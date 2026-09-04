"""Security regressions for Personal Context authority identity binding."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from tldw_profile_core import ProfileRecord
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.core.DB_Management.Personalization_DB import (
    PersonalizationDB,
)
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    IngressIdentity,
    PersonalContextPublicationJournal,
    PersonalContextPublicationRelayStore,
    PublicationObject,
    PublicationRelayLease,
    PublicationSourceRow,
)
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
    PersonalContextAuthorityMetadata,
    PersonalContextExchangeProof,
)
from tldw_Server_API.app.core.Sync.v2.personal_context_relay import PersonalContextRelay
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
from tldw_Server_API.app.core.Sync.v2.store import SyncStoreError, SyncV2Store
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    encoded_master_key,
)


class AuthorityHarness:
    """Real two-store harness for one unacknowledged authority source row."""

    def __init__(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
        self.personal_db = PersonalizationDB.for_path(tmp_path / "personalization.db")
        counters: dict[str, int] = {}

        def next_id(label: str) -> str:
            counters[label] = counters.get(label, 0) + 1
            return f"{label}-{counters[label]}"

        self.canonical = PersonalContextService(
            PersonalContextRepository(self.personal_db),
            clock=lambda: datetime(2026, 9, 3, 12, 0, tzinfo=UTC),
            id_factory=next_id,
        )
        self.manifest = self.canonical.create_profile()
        key_id, integrity_key = self.canonical.sync_integrity_key(
            self.manifest.profile_id
        )
        self.store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db"))
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
        self.service = SyncV2Service(
            store=self.store,
            adapters=adapters,
            personal_context_service_resolver=lambda _user_id: self.canonical,
        )
        self.store.enroll_dataset(
            SyncDatasetCreate(
                dataset_id="dataset-a",
                owner_user_id="user-a",
                encryption_policy="server_trusted_v1",
                domains=list(PERSONAL_CONTEXT_SYNC_DOMAINS),
                metadata={
                    "personal_context": {
                        "profile_id": self.manifest.profile_id,
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
        self.publications = PersonalContextPublicationRelayStore(self.personal_db)
    @contextmanager
    def claimed_row(self) -> Iterator[PublicationSourceRow]:
        with self.publications.profile_lease(self.manifest.profile_id) as lease:
            assert lease is not None
            batch = self.publications.earliest_nonterminal_batch(
                self.manifest.profile_id,
                row_limit=100,
            )
            assert batch is not None
            source = next(item for item in batch.rows if item.row_state != "acknowledged")
            row = replace(source, relay_owner_token=lease.owner_token)
            yield row

    def stage(self, row: PublicationSourceRow) -> int:
        return self.service.stage_personal_context_authority(
            row,
            "dataset-a",
            "user-a",
        )

    def persist_staged_row(self) -> tuple[PublicationSourceRow, int]:
        """Stage the current source and persist its cursor without finalizing it."""

        with self.publications.profile_lease(self.manifest.profile_id) as lease:
            assert lease is not None
            batch = self.publications.earliest_nonterminal_batch(
                self.manifest.profile_id,
                row_limit=100,
            )
            assert batch is not None
            row = replace(batch.rows[0], relay_owner_token=lease.owner_token)
            cursor = self.stage(row)
            self.publications.record_staged_row(
                row,
                server_cursor=cursor,
                lease=lease,
            )
            return row, cursor

    def resume_relay(self):
        """Run the real relay entry point against a previously staged source."""

        return PersonalContextRelay(
            publications=self.publications,
            stage_authority=self.service.stage_personal_context_authority,
            finalize_authority=self.service.finalize_personal_context_authority,
            cancel_authority=self.service.cancel_personal_context_authority,
        ).relay_profile(
            user_id="user-a",
            profile_id=self.manifest.profile_id,
            dataset_id="dataset-a",
            after_server_cursor=None,
            row_budget=1,
            wall_time_ms=5_000,
        )

    def update_sync(self, cursor: int, assignments: str, values: tuple[Any, ...]) -> None:
        with self.store.db.backend.transaction() as connection:
            self.store.db.execute(
                f"UPDATE sync_envelopes SET {assignments} WHERE server_sequence = ?",  # noqa: S608
                (*values, cursor),
                connection=connection,
            )

    def tamper_sync_envelope(self, cursor: int, field: str) -> None:
        """Mutate one persisted immutable fact without changing protected content."""

        direct = {
            "base_server_cursor": ("base_server_cursor = ?", (991,)),
            "base_object_revision": ("base_object_revision = ?", (77,)),
            "base_object_hash": ("base_object_hash = ?", ("sha256:" + "0" * 64,)),
            "object_revision": ("object_revision = ?", (88,)),
            "stable_key": ("stable_key = ?", ("tampered-stable-key",)),
            "client_sequence": ("client_sequence = ?", (991,)),
            "client_timestamp": ("client_timestamp = ?", ("2099-01-01T00:00:00Z",)),
            "client_profile_id": ("client_profile_id = ?", ("tampered-profile",)),
            "originating_device": ("device_id = ?", ("tampered-device",)),
            "client_envelope_id": ("client_envelope_id = ?", ("tampered-envelope",)),
            "encrypted_content": ("payload_ciphertext = ?", ("AAAA",)),
            "dependencies": (
                "dependency_json = ?",
                ('[{"domain":"personal_context.scope","object_id":"tampered"}]',),
            ),
            "mutation_group": (
                "mutation_group_id = ?, mutation_step = ?, mutation_step_count = ?, mutation_plan_hash = ?",
                ("tampered-group", 0, 1, "0" * 64),
            ),
        }
        if field in direct:
            assignments, values = direct[field]
            self.update_sync(cursor, assignments, values)
            return

        with self.store.db.backend.transaction() as connection:
            stored = self.store.db.execute(
                "SELECT routing_metadata_json, encryption_metadata_json "
                "FROM sync_envelopes WHERE server_sequence = ?",
                (cursor,),
                connection=connection,
            ).rows[0]
            routing = json.loads(stored["routing_metadata_json"])
            encryption = json.loads(stored["encryption_metadata_json"])
            if field == "encryption_policy":
                encryption["policy"] = "tampered-policy"
            elif field == "encryption_algorithm":
                encryption["personal_context_at_rest"]["algorithm"] = "tampered"
            elif field == "encryption_key_version":
                encryption["personal_context_at_rest"]["key_version"] = 99
            elif field == "wrapped_dek":
                encryption["personal_context_at_rest"]["wrapped_dek"] = "AAAA"
            elif field == "routing_metadata":
                routing["unexpected_authority_route"] = "tampered"
            elif field == "profile_id":
                routing["profile_id"] = "tampered-profile"
            elif field == "purge_generation":
                routing["purge_generation"] = 99
            elif field == "integrity_key_id":
                routing["integrity_key_id"] = "tampered-key"
            elif field == "authority_tag":
                routing["authority_envelope_tag"] = "hmac-sha256-v1:" + "0" * 64
            elif field == "authority_batch_id":
                routing["personal_context_authority"]["publication_batch_id"] = (
                    "tampered-batch-id"
                )
            elif field == "authority_source_sequence":
                routing["personal_context_authority"]["profile_publication_sequence"] = 99
            elif field == "authority_batch_ordinal":
                routing["personal_context_authority"]["batch_ordinal"] = 99
            elif field == "authority_batch_size":
                routing["personal_context_authority"]["batch_size"] = 99
            else:  # pragma: no cover - parameter list and mutations stay paired.
                raise AssertionError(f"unknown tamper field: {field}")
            self.store.db.execute(
                """UPDATE sync_envelopes
                   SET routing_metadata_json = ?, encryption_metadata_json = ?
                   WHERE server_sequence = ?""",
                (
                    json.dumps(routing, sort_keys=True, separators=(",", ":")),
                    json.dumps(encryption, sort_keys=True, separators=(",", ":")),
                    cursor,
                ),
                connection=connection,
            )

    def source_row_state(self, row: PublicationSourceRow) -> str:
        with self.personal_db.transaction() as connection:
            stored = connection.execute(
                """SELECT row_state FROM personal_context_publication_rows
                   WHERE profile_id = ? AND profile_publication_sequence = ?
                     AND batch_ordinal = ?""",
                (row.profile_id, row.profile_publication_sequence, row.batch_ordinal),
            ).fetchone()
            assert stored is not None
            return str(stored["row_state"])

    def has_attention(self, row: PublicationSourceRow) -> bool:
        with self.personal_db.transaction() as connection:
            stored = connection.execute(
                """SELECT 1 FROM personal_context_publication_relay_attention
                   WHERE profile_id = ? AND profile_publication_sequence = ?""",
                (row.profile_id, row.profile_publication_sequence),
            ).fetchone()
            return stored is not None


class IngressHarness(AuthorityHarness):
    """Two-store harness with a real applied client ingress receipt."""

    def __init__(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__(tmp_path, monkeypatch)
        key_id, integrity_key = self.canonical.sync_integrity_key(
            self.manifest.profile_id
        )
        self.service.materializers = {
            domain: PersonalContextMaterializer(
                domain=domain,
                service_resolver=lambda _user_id: self.canonical,
            )
            for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
        }
        self.service.register_device(
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
        self.store.complete_personal_context_link_receipt(
            user_id="user-a",
            dataset_id="dataset-a",
            device_id="device-a",
            profile_id=self.manifest.profile_id,
            integrity_key_id=key_id,
            purge_generation=0,
            bootstrap_cursor="fixture-cursor",
        )
        self.record = self.canonical.create_manual_record(
            scope_id=self.canonical.list_scopes()[0].scope_id,
            payload={
                "kind": "preference",
                "subject": "response.detail",
                "polarity": "like",
                "value": "concise",
            },
            semantic_key={"namespace": "preference", "subject": "response.detail"},
            controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
        )
        relay_errors: list[str] = []

        def stage_initial(row: PublicationSourceRow, dataset_id: str, user_id: str) -> int:
            try:
                return self.service.stage_personal_context_authority(
                    row, dataset_id, user_id
                )
            except Exception as exc:
                relay_errors.append(f"stage:{type(exc).__name__}:{exc}")
                raise

        def finalize_initial(
            row: PublicationSourceRow,
            cursor: int,
            dataset_id: str,
            user_id: str,
        ) -> None:
            try:
                self.service.finalize_personal_context_authority(
                    row, cursor, dataset_id, user_id
                )
            except Exception as exc:
                relay_errors.append(f"finalize:{type(exc).__name__}:{exc}")
                raise

        relay = PersonalContextRelay(
            publications=self.publications,
            stage_authority=stage_initial,
            finalize_authority=finalize_initial,
            cancel_authority=self.service.cancel_personal_context_authority,
        )
        for _ in range(10):
            initial = relay.relay_profile(
                user_id="user-a",
                profile_id=self.manifest.profile_id,
                dataset_id="dataset-a",
                after_server_cursor=None,
                wall_time_ms=5_000,
            )
            if initial.continuation == "complete":
                break
        assert initial.continuation == "complete", relay_errors
        record_head = self.store.get_current_head(
            "dataset-a", "personal_context.record", self.record.record_id
        )
        assert record_head is not None
        updated = ProfileRecord.model_validate(
            {
                **self.record.model_dump(mode="python"),
                "version_id": "client-record-v2",
                "parent_version_id": self.record.version_id,
                "updated_at": self.record.updated_at + timedelta(seconds=1),
                "payload": {
                    **self.record.payload.model_dump(mode="python"),
                    "value": "structured",
                },
            }
        )
        canonical = canonical_json_bytes(updated.model_dump(mode="json"))
        pushed = self.service.push(
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
                    object_id=updated.record_id,
                    parent_id=updated.scope_id,
                    adapter_version=1,
                    schema_version=1,
                    payload=updated.model_dump(mode="json"),
                    payload_hash="hmac-sha256-v1:"
                    + hmac.new(integrity_key, canonical, hashlib.sha256).hexdigest(),
                    payload_size_bytes=len(canonical),
                    base_server_cursor=record_head.server_cursor,
                    base_object_revision=record_head.object_revision,
                    base_object_hash=record_head.payload_hash,
                    object_revision=(record_head.object_revision or 0) + 1,
                    base_version=self.record.version_id,
                    entity_version=updated.version_id,
                    encryption_metadata={"policy": "server_trusted_v1"},
                    routing_metadata={
                        "integrity_key_id": key_id,
                        "profile_id": self.manifest.profile_id,
                        "purge_generation": 0,
                    },
                )
            ],
            personal_context_exchange=PersonalContextExchangeProof(
                ongoing_sync_version=1,
                activation_epoch="epoch_0123456789abcdef",
                continuity_token="continuity_0123456789abcdef",
            ),
        )
        assert pushed.rejected == []
        assert pushed.conflicts == []
        assert len(pushed.accepted) == 1
        assert pushed.accepted[0].apply_status == "applied", pushed.accepted[0]
        self.ingress_cursor = pushed.accepted[0].server_sequence

    def tamper_sync_receipt(self, column: str, changed: Any) -> None:
        assignments = {
            "device_id": "device_id = ?",
            "client_envelope_id": "client_envelope_id = ?",
            "canonical_payload_digest": "canonical_payload_digest = ?",
            "purge_generation": "purge_generation = ?",
            "resulting_object_id": "resulting_object_id = ?",
            "resulting_internal_version_id": "resulting_internal_version_id = ?",
            "manifest_revision": "manifest_revision = ?",
            "manifest_version_id": "manifest_version_id = ?",
            "publication_batch_id": "publication_batch_id = ?",
            "profile_publication_sequence": "profile_publication_sequence = ?",
            "receipt_id": "receipt_id = ?",
            "wire_entity_version": "wire_entity_version = ?",
        }
        assert column in assignments
        with self.store.db.backend.transaction() as connection:
            self.store.db.execute(
                "UPDATE sync_personal_context_ingress_receipts "
                f"SET {assignments[column]} WHERE server_sequence = ?",  # noqa: S608
                (changed, self.ingress_cursor),
                connection=connection,
            )

    def delete_sync_receipt(self) -> None:
        with self.store.db.backend.transaction() as connection:
            self.store.db.execute(
                """DELETE FROM sync_personal_context_ingress_receipts
                   WHERE server_sequence = ?""",
                (self.ingress_cursor,),
                connection=connection,
            )

    def tamper_canonical_receipt(self, column: str, changed: Any) -> None:
        assignments = {
            "canonical_payload_digest": "canonical_payload_digest = ?",
            "resulting_object_id": "resulting_object_id = ?",
            "resulting_version_id": "resulting_version_id = ?",
            "resulting_manifest_revision": "resulting_manifest_revision = ?",
            "resulting_manifest_version_id": "resulting_manifest_version_id = ?",
            "publication_batch_id": "publication_batch_id = ?",
            "profile_publication_sequence": "profile_publication_sequence = ?",
        }
        assert column in assignments
        with self.personal_db.transaction() as connection:
            connection.execute(
                "UPDATE personal_context_ingress_receipts "
                f"SET {assignments[column]} "  # noqa: S608
                "WHERE dataset_id = ? AND device_id = ? AND client_envelope_id = ?",
                (changed, "dataset-a", "device-a", "device-a:record:v2"),
            )

    def duplicate_canonical_batch_receipt(self) -> None:
        """Add an unrelated receipt that names the same publication result."""

        with self.personal_db.transaction() as connection:
            connection.execute(
                """INSERT INTO personal_context_ingress_receipts(
                       dataset_id, device_id, client_envelope_id,
                       canonical_payload_digest, purge_generation,
                       wire_entity_version, resulting_object_id,
                       resulting_version_id, resulting_manifest_revision,
                       resulting_manifest_version_id, publication_batch_id,
                       profile_publication_sequence, receipt_id, created_at
                   )
                   SELECT 'noise-dataset', 'noise-device', 'noise-envelope',
                          canonical_payload_digest, purge_generation,
                          wire_entity_version, resulting_object_id,
                          resulting_version_id, resulting_manifest_revision,
                          resulting_manifest_version_id, publication_batch_id,
                          profile_publication_sequence, 'noise-receipt', created_at
                   FROM personal_context_ingress_receipts
                   WHERE dataset_id = ? AND device_id = ?
                     AND client_envelope_id = ?""",
                ("dataset-a", "device-a", "device-a:record:v2"),
            )


class PurgeIngressHarness(AuthorityHarness):
    """Real stores containing a receipt-bound, non-materialized client purge."""

    def __init__(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__(tmp_path, monkeypatch)
        relay = PersonalContextRelay(
            publications=self.publications,
            stage_authority=self.service.stage_personal_context_authority,
            finalize_authority=self.service.finalize_personal_context_authority,
            cancel_authority=self.service.cancel_personal_context_authority,
        )
        for _ in range(10):
            result = relay.relay_profile(
                user_id="user-a",
                profile_id=self.manifest.profile_id,
                dataset_id="dataset-a",
                after_server_cursor=None,
                wall_time_ms=5_000,
            )
            if result.continuation == "complete":
                break
        assert result.continuation == "complete"

        payload = {
            "schema_version": 1,
            "profile_id": self.manifest.profile_id,
            "purge_generation": 1,
        }
        canonical = canonical_json_bytes(payload)
        _key_id, integrity_key = self.canonical.sync_integrity_key(
            self.manifest.profile_id
        )
        client = SyncEnvelopeCreate(
            dataset_id="dataset-a",
            client_envelope_id="device-a:purge:1",
            device_id="device-a",
            domain="personal_context.purge",
            operation="tombstone",
            object_id=self.manifest.profile_id,
            object_revision=1,
            schema_version=1,
            adapter_version=1,
            payload=payload,
            payload_hash="hmac-sha256-v1:"
            + hmac.new(integrity_key, canonical, hashlib.sha256).hexdigest(),
            payload_size_bytes=len(canonical),
            entity_version=1,
            deleted=True,
            encryption_metadata={"policy": "server_trusted_v1"},
            routing_metadata={
                "integrity_key_id": _key_id,
                "profile_id": self.manifest.profile_id,
                "purge_generation": 0,
                "personal_context_authority": PersonalContextAuthorityMetadata(
                    role="client_ingress"
                ).model_dump(mode="json"),
            },
        )
        protected = self.service._protect_personal_context_for_storage(
            self.store.get_dataset("dataset-a"),
            client,
        )
        ingress = self.store.insert_envelope(protected)
        assert ingress.server_cursor is not None
        self.ingress_cursor = ingress.server_cursor

        identity = IngressIdentity(
            dataset_id="dataset-a",
            device_id="device-a",
            client_envelope_id="device-a:purge:1",
            canonical_payload_digest="sha256:" + hashlib.sha256(canonical).hexdigest(),
            purge_generation=0,
            wire_entity_version="1",
        )
        keys = self.canonical._repository.key_material_for_test(
            self.manifest.profile_id
        )
        journal = PersonalContextPublicationJournal(keys)
        with self.personal_db.transaction(immediate=True) as connection:
            journal.append_batch(
                connection,
                profile_id=self.manifest.profile_id,
                purge_generation=0,
                objects=(
                    PublicationObject(
                        domain="personal_context.purge",
                        object_id=self.manifest.profile_id,
                        version_id="1",
                        operation="tombstone",
                        role="purge_barrier",
                        canonical=canonical,
                    ),
                    PublicationObject(
                        domain="personal_context.manifest",
                        object_id=self.manifest.profile_id,
                        version_id=self.manifest.current_version_id,
                        operation="upsert",
                        role="manifest",
                        canonical=canonical_json_bytes(
                            self.manifest.model_dump(mode="json")
                        ),
                    ),
                ),
                ingress=identity,
                manifest=self.manifest,
                now="2026-09-03T12:00:00+00:00",
            )
            receipt = connection.execute(
                """SELECT * FROM personal_context_ingress_receipts
                   WHERE dataset_id = ? AND device_id = ? AND client_envelope_id = ?""",
                ("dataset-a", "device-a", "device-a:purge:1"),
            ).fetchone()
        assert receipt is not None
        with self.store.db.backend.transaction() as connection:
            self.store.db.execute(
                """INSERT INTO sync_personal_context_ingress_receipts(
                       server_sequence, dataset_id, device_id, client_envelope_id,
                       canonical_payload_digest, purge_generation, resulting_object_id,
                       resulting_internal_version_id, manifest_revision,
                       manifest_version_id, publication_batch_id,
                       profile_publication_sequence, receipt_id, wire_entity_version
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.ingress_cursor,
                    receipt["dataset_id"],
                    receipt["device_id"],
                    receipt["client_envelope_id"],
                    receipt["canonical_payload_digest"],
                    receipt["purge_generation"],
                    receipt["resulting_object_id"],
                    receipt["resulting_version_id"],
                    receipt["resulting_manifest_revision"],
                    receipt["resulting_manifest_version_id"],
                    receipt["publication_batch_id"],
                    receipt["profile_publication_sequence"],
                    receipt["receipt_id"],
                    receipt["wire_entity_version"],
                ),
                connection=connection,
            )
            self.store.db.execute(
                """UPDATE sync_envelopes SET apply_status = 'applied'
                   WHERE server_sequence = ?""",
                (self.ingress_cursor,),
                connection=connection,
            )


@pytest.fixture
def authority_harness(tmp_path, monkeypatch) -> AuthorityHarness:
    return AuthorityHarness(tmp_path, monkeypatch)


@pytest.fixture
def ingress_harness(tmp_path, monkeypatch) -> IngressHarness:
    return IngressHarness(tmp_path, monkeypatch)


@pytest.fixture
def purge_ingress_harness(tmp_path, monkeypatch) -> PurgeIngressHarness:
    return PurgeIngressHarness(tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "field",
    [
        "base_server_cursor",
        "base_object_revision",
        "base_object_hash",
        "object_revision",
        "stable_key",
        "client_sequence",
        "client_timestamp",
        "client_profile_id",
        "dependencies",
        "mutation_group",
        "encryption_policy",
        "encryption_algorithm",
        "encryption_key_version",
        "wrapped_dek",
        "routing_metadata",
        "profile_id",
        "purge_generation",
        "integrity_key_id",
        "authority_tag",
        "authority_batch_id",
        "authority_source_sequence",
        "authority_batch_ordinal",
        "authority_batch_size",
        "originating_device",
        "client_envelope_id",
    ],
)
def test_existing_authority_row_rejects_any_immutable_drift(
    authority_harness: AuthorityHarness,
    field: str,
) -> None:
    """Every persisted immutable authority fact must be authenticated on reuse."""

    with authority_harness.claimed_row() as row:
        cursor = authority_harness.stage(row)
        authority_harness.tamper_sync_envelope(cursor, field)

        with pytest.raises(SyncStoreError, match="authority receipt is invalid"):
            authority_harness.stage(row)

        assert authority_harness.source_row_state(row) == "pending"
        assert authority_harness.has_attention(row) is False


def test_ingress_confirmation_accepts_exact_cross_store_receipt(
    ingress_harness: IngressHarness,
) -> None:
    """The current ingress head confirms only through its exact canonical receipt."""

    with ingress_harness.claimed_row() as row:
        cursor = ingress_harness.stage(row)
        assert cursor > ingress_harness.ingress_cursor
        assert ingress_harness.stage(row) == cursor
        assert ingress_harness.source_row_state(row) == "pending"


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("device_id", "tampered-device"),
        ("client_envelope_id", "tampered-envelope"),
        ("canonical_payload_digest", "sha256:" + "0" * 64),
        ("purge_generation", 99),
        ("resulting_object_id", "tampered-object"),
        ("resulting_internal_version_id", "tampered-version"),
        ("manifest_revision", 99),
        ("manifest_version_id", "tampered-manifest"),
        ("publication_batch_id", "tampered-batch"),
        ("profile_publication_sequence", 99),
        ("receipt_id", "tampered-receipt"),
        ("wire_entity_version", "tampered-wire-version"),
    ],
)
def test_ingress_confirmation_rejects_tampered_sync_receipt(
    ingress_harness: IngressHarness,
    field: str,
    changed: Any,
) -> None:
    """Canonical confirmation must prove every stored Sync receipt fact."""

    ingress_harness.tamper_sync_receipt(field, changed)
    with ingress_harness.claimed_row() as row:
        with pytest.raises(SyncStoreError, match="authority receipt is invalid"):
            ingress_harness.stage(row)
        assert ingress_harness.source_row_state(row) == "pending"
        assert ingress_harness.has_attention(row) is False


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("canonical_payload_digest", "sha256:" + "0" * 64),
        ("resulting_object_id", "tampered-object"),
        ("resulting_version_id", "tampered-version"),
        ("resulting_manifest_revision", 99),
        ("resulting_manifest_version_id", "tampered-manifest"),
        ("publication_batch_id", "tampered-batch"),
        ("profile_publication_sequence", 99),
    ],
)
def test_ingress_confirmation_rejects_tampered_canonical_receipt(
    ingress_harness: IngressHarness,
    field: str,
    changed: Any,
) -> None:
    """Confirmation fails closed when the canonical receipt identity drifts."""

    ingress_harness.tamper_canonical_receipt(field, changed)
    with ingress_harness.claimed_row() as row:
        with pytest.raises(SyncStoreError, match="authority receipt is invalid"):
            ingress_harness.stage(row)
        assert ingress_harness.source_row_state(row) == "pending"
        assert ingress_harness.has_attention(row) is False


@pytest.mark.parametrize("receipt_store", ["sync", "canonical"])
def test_existing_ingress_authority_reuse_rejects_receipt_drift(
    ingress_harness: IngressHarness,
    receipt_store: str,
) -> None:
    """Deterministic reuse authenticates the ingress receipt used at first stage."""

    with ingress_harness.claimed_row() as row:
        ingress_harness.stage(row)
        if receipt_store == "sync":
            ingress_harness.tamper_sync_receipt(
                "canonical_payload_digest", "sha256:" + "0" * 64
            )
        else:
            ingress_harness.tamper_canonical_receipt(
                "canonical_payload_digest", "sha256:" + "0" * 64
            )

        with pytest.raises(SyncStoreError, match="authority receipt is invalid"):
            ingress_harness.stage(row)
        assert ingress_harness.source_row_state(row) == "pending"
        assert ingress_harness.has_attention(row) is False


@pytest.mark.parametrize(
    "field",
    [
        "base_object_revision",
        "encrypted_content",
        "authority_tag",
    ],
)
def test_relay_resume_reverifies_staged_authority_before_apply(
    authority_harness: AuthorityHarness,
    field: str,
) -> None:
    """A staged source cannot bypass complete authority verification on resume."""

    row, cursor = authority_harness.persist_staged_row()
    authority_harness.tamper_sync_envelope(cursor, field)

    result = authority_harness.resume_relay()

    stored = authority_harness.store.get_envelope_by_server_cursor(cursor)
    assert stored is not None
    assert stored.apply_status == "pending"
    assert authority_harness.source_row_state(row) == "staged"
    assert authority_harness.has_attention(row) is False
    assert result.staged_rows == 0
    assert result.continuation == "personal_context_relay_pending"


@pytest.mark.parametrize("receipt_store", ["sync", "canonical"])
def test_relay_resume_reverifies_staged_ingress_receipt_before_apply(
    ingress_harness: IngressHarness,
    receipt_store: str,
) -> None:
    """A staged ingress authority remains bound to both originating receipts."""

    row, cursor = ingress_harness.persist_staged_row()
    if receipt_store == "sync":
        ingress_harness.tamper_sync_receipt(
            "canonical_payload_digest", "sha256:" + "0" * 64
        )
    else:
        ingress_harness.tamper_canonical_receipt(
            "canonical_payload_digest", "sha256:" + "0" * 64
        )

    result = ingress_harness.resume_relay()

    stored = ingress_harness.store.get_envelope_by_server_cursor(cursor)
    assert stored is not None
    assert stored.apply_status == "pending"
    assert ingress_harness.source_row_state(row) == "staged"
    assert ingress_harness.has_attention(row) is False
    assert result.staged_rows == 0
    assert result.continuation == "personal_context_relay_pending"


@pytest.mark.parametrize("receipt_store", ["sync", "canonical"])
def test_manifest_authority_reuse_rejects_originating_receipt_drift(
    ingress_harness: IngressHarness,
    receipt_store: str,
) -> None:
    """The ingress-derived manifest attests the same receipt as its semantic peer."""

    semantic_row, _semantic_cursor = ingress_harness.persist_staged_row()
    ingress_harness.resume_relay()
    assert ingress_harness.source_row_state(semantic_row) == "acknowledged"

    with ingress_harness.claimed_row() as manifest_row:
        assert manifest_row.role == "manifest"
        ingress_harness.stage(manifest_row)
        if receipt_store == "sync":
            ingress_harness.tamper_sync_receipt(
                "canonical_payload_digest", "sha256:" + "0" * 64
            )
        else:
            ingress_harness.tamper_canonical_receipt(
                "canonical_payload_digest", "sha256:" + "0" * 64
            )

        with pytest.raises(SyncStoreError, match="authority receipt is invalid"):
            ingress_harness.stage(manifest_row)
        assert ingress_harness.source_row_state(manifest_row) == "pending"
        assert ingress_harness.has_attention(manifest_row) is False


def test_exact_client_purge_confirmation_uses_canonical_wire_version(
    purge_ingress_harness: PurgeIngressHarness,
) -> None:
    """Integer purge heads compare to their canonical string receipt version."""

    with purge_ingress_harness.claimed_row() as row:
        assert row.role == "purge_barrier"
        cursor = purge_ingress_harness.stage(row)

        assert cursor > purge_ingress_harness.ingress_cursor
        assert purge_ingress_harness.stage(row) == cursor
        assert purge_ingress_harness.source_row_state(row) == "pending"


def test_authority_receipt_lookup_uses_exact_sync_ingress_identity(
    ingress_harness: IngressHarness,
) -> None:
    """An unrelated same-batch receipt cannot make exact confirmation ambiguous."""

    ingress_harness.duplicate_canonical_batch_receipt()
    with ingress_harness.claimed_row() as row:
        assert row.role == "semantic"
        cursor = ingress_harness.stage(row)

        assert cursor > ingress_harness.ingress_cursor
        assert ingress_harness.source_row_state(row) == "pending"


@pytest.mark.parametrize(
    "receipt_state",
    ["missing_sync", "sync_mismatch", "canonical_mismatch"],
)
def test_manifest_prestage_requires_complete_matching_origin_receipt(
    ingress_harness: IngressHarness,
    receipt_state: str,
) -> None:
    """A companion manifest cannot sign an absent or mismatched origin receipt."""

    semantic_row, _semantic_cursor = ingress_harness.persist_staged_row()
    ingress_harness.resume_relay()
    assert ingress_harness.source_row_state(semantic_row) == "acknowledged"

    if receipt_state == "missing_sync":
        ingress_harness.delete_sync_receipt()
    elif receipt_state == "sync_mismatch":
        ingress_harness.tamper_sync_receipt(
            "canonical_payload_digest", "sha256:" + "0" * 64
        )
    else:
        ingress_harness.tamper_canonical_receipt(
            "canonical_payload_digest", "sha256:" + "0" * 64
        )

    error: SyncStoreError | None = None
    with ingress_harness.claimed_row() as manifest_row:
        assert manifest_row.role == "manifest"
        lease = PublicationRelayLease(
            manifest_row.profile_id,
            str(manifest_row.relay_owner_token),
        )
        try:
            cursor = ingress_harness.stage(manifest_row)
        except SyncStoreError as exc:
            error = exc
        else:
            ingress_harness.publications.record_staged_row(
                manifest_row,
                server_cursor=cursor,
                lease=lease,
            )
            ingress_harness.service.finalize_personal_context_authority(
                manifest_row,
                cursor,
                "dataset-a",
                "user-a",
            )
            ingress_harness.publications.acknowledge_row(
                manifest_row,
                server_cursor=cursor,
                lease=lease,
            )

        stored = ingress_harness.store.get_envelope_by_client_id(
            "dataset-a",
            manifest_row.deterministic_envelope_id,
        )
        assert ingress_harness.source_row_state(manifest_row) == "pending"
        assert stored is None
        assert isinstance(error, SyncStoreError)
        assert str(error) == "Personal Context authority receipt is invalid"
        assert ingress_harness.has_attention(manifest_row) is False
