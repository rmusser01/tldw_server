"""Regression coverage for ordered Personal Context publication relay."""

from __future__ import annotations

import hashlib
import hmac
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from tldw_profile_core import ProfileRecord
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    AuthorityStageReceipt,
    PersonalContextPublicationRelayStore,
    PublicationSourceBatch,
    PublicationSourceRow,
)


def test_relay_bounds_source_decryption_before_loading_a_batch() -> None:
    """The source fetch receives the remaining total inspection budget."""

    class BoundedSource:
        @contextmanager
        def profile_lease(self, _profile_id):
            yield SimpleNamespace(owner_token="owner")

        def earliest_nonterminal_batch(
            self, _profile_id, *, row_limit, lease=None, budget=None
        ):
            del lease
            assert row_limit == 7
            assert budget is not None
            return None

    from tldw_Server_API.app.core.Sync.v2.personal_context_relay import (
        PersonalContextRelay,
    )

    result = PersonalContextRelay(
        publications=BoundedSource(),
        stage_authority=lambda *_args: 1,
    ).relay_profile(
        user_id="user-a",
        profile_id="profile-a",
        dataset_id="dataset-a",
        after_server_cursor=None,
        row_budget=7,
    )

    assert result.continuation == "complete"


def test_real_relay_extends_current_sync_heads_across_publication_batches(
    tmp_path, monkeypatch
) -> None:
    """A later manifest must CAS the exact Sync head instead of restarting at one."""

    from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
    from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
    from tldw_Server_API.app.core.Personalization.personal_context_repository import (
        PersonalContextRepository,
    )
    from tldw_Server_API.app.core.Personalization.personal_context_service import (
        PersonalContextService,
    )
    from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
    from tldw_Server_API.app.core.Sync.v2.domain_adapters.personal_context import (
        PersonalContextDomainAdapter,
    )
    from tldw_Server_API.app.core.Sync.v2.materializers.personal_context import (
        PersonalContextMaterializer,
    )
    from tldw_Server_API.app.core.Sync.v2.models import (
        PERSONAL_CONTEXT_SYNC_DOMAINS,
        SyncDatasetCreate,
        SyncEnvelopeCreate,
    )
    from tldw_Server_API.app.core.Sync.v2.personal_context_ongoing_contract import (
        PersonalContextExchangeProof,
    )
    from tldw_Server_API.app.core.Sync.v2.personal_context_relay import (
        PersonalContextAuthoritySourceError,
        PersonalContextRelay,
    )
    from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
    from tldw_Server_API.app.core.Sync.v2.store import SyncStoreError, SyncV2Store
    from tldw_Server_API.tests.Personalization.personal_context_test_support import (
        encoded_master_key,
    )

    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
    personal_db = PersonalizationDB.for_path(tmp_path / "personalization.db")
    counters: dict[str, int] = {}

    def next_id(label: str) -> str:
        counters[label] = counters.get(label, 0) + 1
        return f"{label}-{counters[label]}"

    canonical = PersonalContextService(
        PersonalContextRepository(personal_db),
        clock=lambda: datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
        id_factory=next_id,
    )
    manifest = canonical.create_profile()
    record = canonical.create_manual_record(
        scope_id=canonical.list_scopes()[0].scope_id,
        payload={
            "kind": "preference",
            "subject": "response.detail",
            "polarity": "like",
            "value": "concise",
        },
        semantic_key={"namespace": "preference", "subject": "response.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    key_id, integrity_key = canonical.sync_integrity_key(manifest.profile_id)
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db"))
    adapters = SyncAdapterRegistry(
        [
            PersonalContextDomainAdapter(
                domain=domain,
                integrity_key_resolver=lambda _dataset, _key_id: integrity_key,
                encryption_key_resolver=lambda _dataset: (b"e" * 32, 1),
            )
            for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
        ]
    )
    service = SyncV2Service(
        store=store,
        adapters=adapters,
        materializers={
            domain: PersonalContextMaterializer(
                domain=domain,
                service_resolver=lambda _user_id: canonical,
            )
            for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
        },
        personal_context_service_resolver=lambda _user_id: canonical,
    )
    service.register_device(
        user_id="user-a",
        display_name="device-a",
        client_type="chatbook",
        device_id="device-a",
        capabilities={
            "supported_adapter_versions": {
                domain: [1] for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
            }
        },
    )
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-a",
            owner_user_id="user-a",
            encryption_policy="server_trusted_v1",
            domains=list(PERSONAL_CONTEXT_SYNC_DOMAINS),
            metadata={
                "personal_context": {
                    "profile_id": manifest.profile_id,
                    "integrity_key_id": key_id,
                    "purge_generation": 0,
                    "link_state": "complete",
                    "ongoing_sync_version": 1,
                    "activation_epoch": "epoch_0123456789abcdef",
                    "continuity_token": "continuity_0123456789abcdef",
                }
            },
        )
    )
    store.complete_personal_context_link_receipt(
        user_id="user-a",
        dataset_id="dataset-a",
        device_id="device-a",
        profile_id=manifest.profile_id,
        integrity_key_id=key_id,
        purge_generation=0,
        bootstrap_cursor="fixture-cursor",
    )
    staging_errors: list[str] = []
    finalization_errors: list[str] = []

    def stage(row, dataset_id: str, user_id: str) -> int:
        try:
            return service.stage_personal_context_authority(row, dataset_id, user_id)
        except Exception as exc:
            staging_errors.append(f"{row.domain}:{row.role}:{exc!r}")
            raise

    def finalize(row, server_cursor: int, dataset_id: str, user_id: str) -> None:
        try:
            service.finalize_personal_context_authority(
                row, server_cursor, dataset_id, user_id
            )
        except Exception as exc:
            finalization_errors.append(f"{row.domain}:{row.role}:{exc!r}")
            raise

    publications = PersonalContextPublicationRelayStore(personal_db)
    with publications.profile_lease(manifest.profile_id) as binding_lease:
        assert binding_lease is not None
        source_batch = publications.earliest_nonterminal_batch(
            manifest.profile_id,
            row_limit=100,
        )
        assert source_batch is not None
        source_row = replace(
            source_batch.rows[0], relay_owner_token=binding_lease.owner_token
        )
        invalid_bindings = (
            (
                replace(
                    source_row,
                    integrity_tag="hmac-sha256-v1:" + "0" * 64,
                ),
                "dataset-a",
                PersonalContextAuthoritySourceError,
            ),
            (
                replace(
                    source_row,
                    deterministic_envelope_id="wrong-profile-envelope",
                    profile_id="profile-other",
                ),
                "dataset-a",
                SyncStoreError,
            ),
            (
                replace(
                    source_row,
                    deterministic_envelope_id="wrong-generation-envelope",
                    purge_generation=1,
                ),
                "dataset-a",
                SyncStoreError,
            ),
            (
                replace(
                    source_row,
                    deterministic_envelope_id="wrong-owner-envelope",
                    relay_owner_token="not-the-owner",
                ),
                "dataset-a",
                SyncStoreError,
            ),
            (
                replace(
                    source_row,
                    deterministic_envelope_id="wrong-dataset-envelope",
                ),
                "dataset-other",
                SyncStoreError,
            ),
        )
        for invalid_row, invalid_dataset, error_type in invalid_bindings:
            with pytest.raises(error_type):
                service.stage_personal_context_authority(
                    invalid_row, invalid_dataset, "user-a"
                )
            assert store.get_envelope_by_client_id(
                invalid_dataset, invalid_row.deterministic_envelope_id
            ) is None
        for key_result, error_type in (
            (("wrong-key-id", integrity_key), SyncStoreError),
            (
                ((key_id, b"x" * 32)),
                PersonalContextAuthoritySourceError,
            ),
        ):
            wrong_key_service = SyncV2Service(
                store=store,
                adapters=adapters,
                personal_context_service_resolver=lambda _user_id, result=key_result: SimpleNamespace(
                    _repository=canonical._repository,
                    sync_integrity_key=lambda _profile_id: result,
                ),
            )
            wrong_key_row = source_row
            with pytest.raises(error_type):
                wrong_key_service.stage_personal_context_authority(
                    wrong_key_row, "dataset-a", "user-a"
                )
            assert store.get_envelope_by_client_id(
                "dataset-a", wrong_key_row.deterministic_envelope_id
            ) is None
        cancelled_receipt = service.stage_personal_context_authority(
            source_row, "dataset-a", "user-a"
        )
        cancelled = store.get_envelope_by_server_cursor(
            cancelled_receipt.server_cursor
        )
        assert cancelled is not None
        assert cancelled.apply_status == "pending"
        with store.db.backend.transaction() as connection:
            store.db.execute(
                "UPDATE sync_envelopes SET schema_version = 2 "
                "WHERE server_sequence = ?",
                (cancelled_receipt.server_cursor,),
                connection=connection,
            )
        with pytest.raises(SyncStoreError, match="authority receipt is invalid"):
            service.stage_personal_context_authority(
                source_row,
                "dataset-a",
                "user-a",
            )
        service.cancel_personal_context_authority(
            source_row,
            cancelled_receipt,
            "dataset-a",
            "user-a",
        )
        assert (
            store.get_envelope_by_server_cursor(cancelled_receipt.server_cursor)
            is None
        )
        assert store.get_current_head(
            "dataset-a", source_row.domain, source_row.object_id
        ) is None
    expired_receipt: AuthorityStageReceipt | None = None

    def expire_after_stage(row, dataset_id: str, user_id: str) -> AuthorityStageReceipt:
        nonlocal expired_receipt
        expired_receipt = service.stage_personal_context_authority(
            row,
            dataset_id,
            user_id,
        )
        with personal_db.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE personal_context_publication_relay_leases "
                "SET expires_at_ns = 0 WHERE profile_id = ?",
                (manifest.profile_id,),
            )
        return expired_receipt

    expired = PersonalContextRelay(
        publications=PersonalContextPublicationRelayStore(personal_db),
        stage_authority=expire_after_stage,
        finalize_authority=finalize,
        cancel_authority=service.cancel_personal_context_authority,
    ).relay_profile(
        user_id="user-a",
        profile_id=manifest.profile_id,
        dataset_id="dataset-a",
        after_server_cursor=None,
        row_budget=1,
    )
    assert expired.continuation == "personal_context_relay_pending"
    assert expired_receipt is not None
    expired_staged = store.get_envelope_by_server_cursor(
        expired_receipt.server_cursor
    )
    assert expired_staged is not None
    assert expired_staged.apply_status == "pending"

    original_acknowledge = publications.acknowledge_row
    interrupted = True

    def interrupt_after_sync_insert(row, *, server_cursor: int, lease) -> None:
        nonlocal interrupted
        if interrupted:
            interrupted = False
            raise RuntimeError("crash after durable Sync insert")
        original_acknowledge(row, server_cursor=server_cursor, lease=lease)

    publications.acknowledge_row = interrupt_after_sync_insert
    relay = PersonalContextRelay(
        publications=publications,
        stage_authority=stage,
        finalize_authority=finalize,
        cancel_authority=service.cancel_personal_context_authority,
    )

    interrupted_result = relay.relay_profile(
        user_id="user-a",
        profile_id=manifest.profile_id,
        dataset_id="dataset-a",
        after_server_cursor=None,
    )
    assert interrupted_result.continuation == "personal_context_relay_pending"
    scope_head = store.get_current_head(
        "dataset-a", "personal_context.scope", canonical.list_scopes()[0].scope_id
    )
    assert scope_head is not None
    assert scope_head.apply_status == "pending"

    publications = PersonalContextPublicationRelayStore(personal_db)
    relay = PersonalContextRelay(
        publications=publications,
        stage_authority=stage,
        finalize_authority=finalize,
        cancel_authority=service.cancel_personal_context_authority,
    )
    result = relay.relay_profile(
        user_id="user-a",
        profile_id=manifest.profile_id,
        dataset_id="dataset-a",
        after_server_cursor=None,
        wall_time_ms=5_000,
    )

    assert result.continuation == "complete", staging_errors
    manifest_head = store.get_current_head(
        "dataset-a", "personal_context.manifest", manifest.profile_id
    )
    assert manifest_head is not None
    assert manifest_head.object_revision == 2
    assert manifest_head.base_server_cursor is not None

    record_head = store.get_current_head(
        "dataset-a", "personal_context.record", record.record_id
    )
    assert record_head is not None
    updated = ProfileRecord.model_validate(
        {
            **record.model_dump(mode="python"),
            "version_id": "client-record-v2",
            "parent_version_id": record.version_id,
            "updated_at": record.updated_at + timedelta(seconds=1),
            "payload": {
                **record.payload.model_dump(mode="python"),
                "value": "structured",
            },
        }
    )
    updated_payload = updated.model_dump(mode="json")
    updated_canonical = canonical_json_bytes(updated_payload)
    exchange = PersonalContextExchangeProof(
        ongoing_sync_version=1,
        activation_epoch="epoch_0123456789abcdef",
        continuity_token="continuity_0123456789abcdef",
    )
    pushed = service.push(
        user_id="user-a",
        dataset_id="dataset-a",
        device_id="device-a",
        envelopes=[
            SyncEnvelopeCreate(
                dataset_id="dataset-a",
                client_envelope_id="device-a:record:v2",
                device_id="device-a",
                domain="personal_context.record",
                operation="upsert",
                object_id=record.record_id,
                parent_id=record.scope_id,
                adapter_version=1,
                schema_version=1,
                payload=updated_payload,
                payload_hash="hmac-sha256-v1:"
                + hmac.new(integrity_key, updated_canonical, hashlib.sha256).hexdigest(),
                payload_size_bytes=len(updated_canonical),
                base_server_cursor=record_head.server_cursor,
                base_object_revision=record_head.object_revision,
                base_object_hash=record_head.payload_hash,
                object_revision=(record_head.object_revision or 0) + 1,
                base_version=record.version_id,
                entity_version=updated.version_id,
                encryption_metadata={"policy": "server_trusted_v1"},
                routing_metadata={
                    "integrity_key_id": key_id,
                    "profile_id": manifest.profile_id,
                    "purge_generation": 0,
                },
            )
        ],
        personal_context_exchange=exchange,
    )
    assert pushed.rejected == []
    assert pushed.conflicts == []
    assert len(pushed.accepted) == 1

    confirmation = relay.relay_profile(
        user_id="user-a",
        profile_id=manifest.profile_id,
        dataset_id="dataset-a",
        after_server_cursor=None,
        wall_time_ms=5_000,
    )

    assert confirmation.continuation == "complete", (staging_errors, finalization_errors)
    confirmed_head = store.get_current_head(
        "dataset-a", "personal_context.record", record.record_id
    )
    assert confirmed_head is not None
    assert confirmed_head.authority is not None
    assert confirmed_head.authority.role == "home_authority"
    assert confirmed_head.entity_version == updated.version_id
    assert confirmed_head.apply_status == "applied"


def test_relay_exports_a_result_type() -> None:
    """The relay boundary is available independently of HTTP transport."""

    from tldw_Server_API.app.core.Sync.v2.personal_context_relay import (
        PersonalContextRelayResult,
    )

    assert PersonalContextRelayResult.__name__ == "PersonalContextRelayResult"


def test_corrupt_source_persists_content_free_attention_after_rollback(
    tmp_path, monkeypatch
) -> None:
    """Authenticated source corruption must still block after service reconstruction."""

    from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
    from tldw_Server_API.app.core.Personalization.personal_context_repository import (
        PersonalContextRepository,
    )
    from tldw_Server_API.app.core.Personalization.personal_context_service import (
        PersonalContextService,
    )
    from tldw_Server_API.app.core.Sync.v2.personal_context_relay import (
        PersonalContextRelay,
    )
    from tldw_Server_API.tests.Personalization.personal_context_test_support import (
        encoded_master_key,
    )

    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
    database = PersonalizationDB.for_path(tmp_path / "poison.db")
    canonical = PersonalContextService(PersonalContextRepository(database))
    manifest = canonical.create_profile()
    with database.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE personal_context_publication_rows SET ciphertext = ? "
            "WHERE profile_id = ? AND profile_publication_sequence = 1 "
            "AND batch_ordinal = 0",
            (b"corrupt", manifest.profile_id),
        )

    first = PersonalContextRelay(
        publications=PersonalContextPublicationRelayStore(database),
        stage_authority=lambda *_args: pytest.fail("corrupt source reached staging"),
    ).relay_profile(
        user_id="user-a",
        profile_id=manifest.profile_id,
        dataset_id="dataset-a",
        after_server_cursor=None,
    )
    second = PersonalContextRelay(
        publications=PersonalContextPublicationRelayStore(database),
        stage_authority=lambda *_args: pytest.fail("poisoned source reached staging"),
    ).relay_profile(
        user_id="user-a",
        profile_id=manifest.profile_id,
        dataset_id="dataset-a",
        after_server_cursor=None,
    )

    assert first.continuation == "relay_poisoned"
    assert second.continuation == "relay_poisoned"
    with database.transaction() as connection:
        attention = connection.execute(
            "SELECT error_code FROM personal_context_publication_relay_attention "
            "WHERE profile_id = ? AND profile_publication_sequence = 1",
            (manifest.profile_id,),
        ).fetchone()
    assert attention is not None
    assert tuple(attention) == ("relay_poisoned",)


class _Publications:
    def __init__(self) -> None:
        self.rows = [
            _row(0, "semantic"),
            _row(1, "manifest"),
        ]
        self.acknowledged: list[int] = []
        self.attention = False
        self.failed_after: int | None = None

    @contextmanager
    def profile_lease(self, _profile_id: str):
        yield type("Lease", (), {"owner_token": "lease-token"})()

    def renew_lease(self, _lease: object) -> bool:
        return True

    def row_is_current(self, _row: PublicationSourceRow, _lease: object) -> bool:
        return True

    def earliest_nonterminal_batch(
        self,
        _profile_id: str,
        *,
        row_limit: int,
        lease: object | None = None,
        budget: object | None = None,
    ) -> PublicationSourceBatch:
        del lease
        assert row_limit > 0
        assert budget is not None
        if all(row.row_state == "acknowledged" for row in self.rows):
            return None  # type: ignore[return-value]
        selected: list[PublicationSourceRow] = []
        for row in self.rows[:row_limit]:
            if not budget.consume():
                break
            selected.append(row)
        return PublicationSourceBatch(
            "profile-a",
            1,
            "batch-a",
            tuple(selected),
        )

    def acknowledge_row(self, row: PublicationSourceRow, *, server_cursor: int, lease: object) -> None:
        del lease
        if self.failed_after is not None and row.batch_ordinal == self.failed_after:
            raise RuntimeError("injected interruption")
        self.acknowledged.append(server_cursor)
        self.rows[row.batch_ordinal] = replace(row, row_state="acknowledged", sync_server_cursor=server_cursor)

    def record_staged_row(
        self, row: PublicationSourceRow, *, server_cursor: int, lease: object
    ) -> None:
        assert lease.owner_token == "lease-token"
        self.rows[row.batch_ordinal] = replace(
            row,
            row_state="staged",
            sync_server_cursor=server_cursor,
        )

    def complete_if_acknowledged(
        self, _batch: PublicationSourceBatch, *, lease: object
    ) -> bool:
        assert lease.owner_token == "lease-token"
        return all(row.row_state == "acknowledged" for row in self.rows)

    def mark_attention(
        self, _batch: PublicationSourceBatch, *, lease: object
    ) -> None:
        assert lease.owner_token == "lease-token"
        self.attention = True


def _row(ordinal: int, role: str) -> PublicationSourceRow:
    return PublicationSourceRow(
        profile_id="profile-a", profile_publication_sequence=1, publication_batch_id="batch-a",
        batch_ordinal=ordinal, batch_size=2, purge_generation=0, role=role,  # type: ignore[arg-type]
        object_id=f"object-{ordinal}", version_id=f"version-{ordinal}", operation="upsert",
        deterministic_envelope_id=f"envelope-{ordinal}", domain="personal_context.record",
        canonical=b'{}', integrity_tag="hmac-sha256-v1:" + "a" * 64,
        sync_server_cursor=None, row_state="pending",
    )


def _stage_receipt(row: PublicationSourceRow, cursor: int) -> AuthorityStageReceipt:
    return AuthorityStageReceipt(
        server_cursor=cursor,
        deterministic_envelope_id=row.deterministic_envelope_id,
        publication_batch_id=row.publication_batch_id,
        profile_publication_sequence=row.profile_publication_sequence,
        batch_ordinal=row.batch_ordinal,
        batch_size=row.batch_size,
        purge_generation=row.purge_generation,
    )


def test_relay_never_stages_manifest_before_semantic_siblings() -> None:
    from tldw_Server_API.app.core.Sync.v2.personal_context_relay import PersonalContextRelay

    publications = _Publications()
    staged: list[str] = []
    relay = PersonalContextRelay(
        publications=publications,
        stage_authority=lambda row, _dataset, _user: (
            staged.append(row.role) or _stage_receipt(row, row.batch_ordinal + 1)
        ),
    )

    publications.failed_after = 0
    first = relay.relay_profile(
        user_id="user-a",
        profile_id="profile-a",
        dataset_id="dataset-a",
        after_server_cursor=None,
    )
    assert first.continuation == "personal_context_relay_pending"
    assert staged == ["semantic"]
    publications.failed_after = None
    relay.relay_profile(user_id="user-a", profile_id="profile-a", dataset_id="dataset-a", after_server_cursor=None)
    assert staged == ["semantic", "manifest"]


def test_relay_poison_blocks_the_earliest_batch_without_error_body() -> None:
    from tldw_Server_API.app.core.Sync.v2.personal_context_relay import (
        PersonalContextAuthoritySourceError,
        PersonalContextRelay,
    )

    publications = _Publications()
    relay = PersonalContextRelay(
        publications=publications,
        stage_authority=lambda *_args: (_ for _ in ()).throw(
            PersonalContextAuthoritySourceError("secret")
        ),
    )

    result = relay.relay_profile(
        user_id="user-a", profile_id="profile-a", dataset_id="dataset-a", after_server_cursor=None
    )

    assert result.continuation == "relay_poisoned"
    assert publications.attention is True
