"""Security regressions for Personal Context authority identity binding."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
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
    RecordMutation,
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
    preference_record,
)

pytestmark = pytest.mark.unit


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
        # These tests intentionally retain unacknowledged source publications.
        # Isolate activation authorization instead of covering their test subjects;
        # the activation integration suite verifies real cross-store receipts.
        monkeypatch.setattr(
            self.canonical._repository,
            "validate_activation_exchange",
            self.validate_activation_exchange,
        )
        key_id, integrity_key = self.canonical.sync_integrity_key(
            self.manifest.profile_id
        )
        self.integrity_key = integrity_key
        self.integrity_key_id = key_id
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

    def validate_activation_exchange(
        self,
        *,
        profile_id: str,
        device_id: str,
        dataset_id: str,
        activation_epoch: str,
        continuity_token: str,
    ) -> PersonalContextExchangeProof:
        """Authenticate this fixture's fixed proof independently of Sync metadata."""
        proof = PersonalContextExchangeProof(
            ongoing_sync_version=1,
            activation_epoch=activation_epoch,
            continuity_token=continuity_token,
        )
        if (
            profile_id != self.manifest.profile_id
            or device_id != "device-a"
            or dataset_id != "dataset-a"
            or proof.activation_epoch != "epoch_0123456789abcdef"
            or proof.continuity_token != "continuity_0123456789abcdef"
        ):
            raise ValueError("personal_context_activation_required")
        return proof

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
        ).server_cursor

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

    def resume_relay(self, *, row_budget: int = 1):
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
            row_budget=row_budget,
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


class FirstIngressHarness(AuthorityHarness):
    """Real stores whose first record ingress omits its optional wire revision."""

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
        relay = PersonalContextRelay(
            publications=self.publications,
            stage_authority=self.service.stage_personal_context_authority,
            finalize_authority=self.service.finalize_personal_context_authority,
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
        assert initial.continuation == "complete"

        seed = preference_record(
            self.manifest.profile_id,
            record_id="first-client-record",
            version_id="first-client-record-v1",
        )
        self.record = ProfileRecord.model_validate(
            {
                **seed.model_dump(mode="python"),
                "scope_id": self.canonical.list_scopes()[0].scope_id,
            }
        )
        canonical = canonical_json_bytes(self.record.model_dump(mode="json"))
        pushed = self.service.push(
            user_id="user-a",
            dataset_id="dataset-a",
            device_id="device-a",
            envelopes=[
                SyncEnvelopeCreate(
                    dataset_id="dataset-a",
                    client_envelope_id="device-a:first-record",
                    device_id="device-a",
                    domain="personal_context.record",
                    operation="upsert",
                    object_id=self.record.record_id,
                    parent_id=self.record.scope_id,
                    adapter_version=1,
                    schema_version=1,
                    payload=self.record.model_dump(mode="json"),
                    payload_hash="hmac-sha256-v1:"
                    + hmac.new(integrity_key, canonical, hashlib.sha256).hexdigest(),
                    payload_size_bytes=len(canonical),
                    entity_version=self.record.version_id,
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
        self.ingress_cursor = pushed.accepted[0].server_sequence
        self.ingress = self.store.get_envelope_by_server_cursor(self.ingress_cursor)
        assert self.ingress is not None
        assert self.ingress.object_revision is None
        self.projected_state = self.store.get_object_state(
            "dataset-a", "personal_context.record", self.record.record_id
        )
        assert self.projected_state is not None
        assert self.projected_state.dataset_id == self.ingress.dataset_id
        assert self.projected_state.domain == self.ingress.domain
        assert self.projected_state.object_id == self.ingress.object_id
        assert self.projected_state.object_revision == 1
        assert self.projected_state.latest_server_cursor == self.ingress_cursor
        assert self.projected_state.object_hash == self.ingress.payload_hash
        assert self.projected_state.deleted == self.ingress.deleted

    def pending_rows(self) -> tuple[PublicationSourceRow, ...]:
        batch = self.publications.earliest_nonterminal_batch(
            self.manifest.profile_id,
            row_limit=100,
        )
        assert batch is not None
        return batch.rows

    def corrupt_projected_state(self, case: str) -> None:
        if case == "missing":
            with self.store.db.backend.transaction() as connection:
                self.store.db.execute(
                    "DELETE FROM sync_object_state WHERE dataset_id = ? "
                    "AND domain = ? AND object_id = ?",
                    ("dataset-a", "personal_context.record", self.record.record_id),
                    connection=connection,
                )
            return
        assignments = {
            "latest_cursor": ("latest_server_cursor = ?", self.ingress_cursor + 1),
            "object_hash": ("object_hash = ?", "hmac-sha256-v1:" + "0" * 64),
            "deleted": ("deleted = ?", 1),
        }
        assignment, value = assignments[case]
        with self.store.db.backend.transaction() as connection:
            self.store.db.execute(
                f"UPDATE sync_object_state SET {assignment} "  # noqa: S608
                "WHERE dataset_id = ? AND domain = ? AND object_id = ?",
                (value, "dataset-a", "personal_context.record", self.record.record_id),
                connection=connection,
            )

    def push_omitted_revision_update(self) -> tuple[int, ProfileRecord]:
        """Apply a real update whose wire revision is omitted but base is complete."""

        head = self.store.get_current_head(
            "dataset-a", "personal_context.record", self.record.record_id
        )
        assert head is not None
        assert head.object_revision is not None
        updated = ProfileRecord.model_validate(
            {
                **self.record.model_dump(mode="python"),
                "version_id": "first-client-record-v2",
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
                    client_envelope_id="device-a:first-record:v2",
                    device_id="device-a",
                    domain="personal_context.record",
                    operation="upsert",
                    object_id=updated.record_id,
                    parent_id=updated.scope_id,
                    adapter_version=1,
                    schema_version=1,
                    payload=updated.model_dump(mode="json"),
                    payload_hash="hmac-sha256-v1:"
                    + hmac.new(
                        self.integrity_key, canonical, hashlib.sha256
                    ).hexdigest(),
                    payload_size_bytes=len(canonical),
                    base_server_cursor=head.server_cursor,
                    base_object_revision=head.object_revision,
                    base_object_hash=head.payload_hash,
                    base_version=self.record.version_id,
                    entity_version=updated.version_id,
                    encryption_metadata={"policy": "server_trusted_v1"},
                    routing_metadata={
                        "integrity_key_id": self.integrity_key_id,
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
        stored = self.store.get_envelope_by_server_cursor(
            pushed.accepted[0].server_sequence
        )
        assert stored is not None
        assert stored.object_revision is None
        assert pushed.accepted[0].apply_status == "applied"
        return pushed.accepted[0].server_sequence, updated


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
def authority_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AuthorityHarness:
    """Create isolated canonical and Sync stores with an authority binding."""

    return AuthorityHarness(tmp_path, monkeypatch)


@pytest.fixture
def ingress_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> IngressHarness:
    """Create an authority harness with an existing client-ingress receipt."""

    return IngressHarness(tmp_path, monkeypatch)


@pytest.fixture
def first_ingress_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> FirstIngressHarness:
    """Create a harness for the first client-ingress publication."""

    return FirstIngressHarness(tmp_path, monkeypatch)


@pytest.fixture
def purge_ingress_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> PurgeIngressHarness:
    """Create an authority harness carrying an incoming client purge publication."""

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


def test_first_ingress_without_wire_revision_relays_semantic_and_manifest(
    first_ingress_harness: FirstIngressHarness,
) -> None:
    """Projected revision 1 authenticates both successors without mutating ingress."""

    rows = first_ingress_harness.pending_rows()
    assert [row.role for row in rows] == ["semantic", "manifest"]

    for _ in range(10):
        result = first_ingress_harness.resume_relay(row_budget=100)
        if result.continuation == "complete":
            break

    assert result.continuation == "complete"
    assert all(
        first_ingress_harness.source_row_state(row) == "acknowledged" for row in rows
    )
    unchanged = first_ingress_harness.store.get_envelope_by_server_cursor(
        first_ingress_harness.ingress_cursor
    )
    assert unchanged is not None
    assert unchanged.object_revision is None
    for domain, object_id in (
        ("personal_context.record", first_ingress_harness.record.record_id),
        ("personal_context.manifest", first_ingress_harness.manifest.profile_id),
    ):
        authority = first_ingress_harness.store.get_current_head(
            "dataset-a", domain, object_id
        )
        assert authority is not None
        assert authority.base_object_revision == 1
        assert authority.object_revision == 2
        assert authority.authority is not None
        assert authority.authority.role == "home_authority"


@pytest.mark.parametrize("case", ["missing", "latest_cursor", "object_hash", "deleted"])
def test_first_ingress_projection_tamper_does_not_change_immutable_lineage(
    first_ingress_harness: FirstIngressHarness,
    case: str,
) -> None:
    """A durable ingress receipt, rather than mutable latest projection, proves lineage."""

    first_ingress_harness.corrupt_projected_state(case)
    with first_ingress_harness.claimed_row() as row:
        assert row.role == "semantic"
        cursor = first_ingress_harness.stage(row)
        authority = first_ingress_harness.store.get_envelope_by_server_cursor(cursor)
        assert authority is not None
        assert authority.base_object_revision == 1
        assert authority.object_revision == 2
        assert (
            first_ingress_harness.store.get_envelope_by_client_id(
                "dataset-a", row.deterministic_envelope_id
            )
            is not None
        )
        assert first_ingress_harness.source_row_state(row) == "pending"


def test_first_ingress_authority_stage_never_reads_projection(
    first_ingress_harness: FirstIngressHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authority staging spends no hidden row on latest projection state."""

    def forbidden_projection_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("authority lineage must not read sync_object_state")

    monkeypatch.setattr(SyncV2Store, "get_object_state", forbidden_projection_read)
    with first_ingress_harness.claimed_row() as row:
        assert row.role == "semantic"
        cursor = first_ingress_harness.stage(row)
        assert cursor > first_ingress_harness.ingress_cursor
        assert first_ingress_harness.source_row_state(row) == "pending"


@pytest.mark.parametrize("malformed", [1.5, "not-an-integer", float("inf")])
def test_personal_context_authority_cas_rejects_malformed_raw_revision(
    first_ingress_harness: FirstIngressHarness,
    malformed: object,
) -> None:
    """Storage affinity coercion cannot turn a malformed raw revision into authority."""

    first_ingress_harness.update_sync(
        first_ingress_harness.ingress_cursor,
        "object_revision = ?",
        (malformed,),
    )
    with first_ingress_harness.claimed_row() as row:
        with pytest.raises(SyncStoreError):
            first_ingress_harness.stage(row)
        assert first_ingress_harness.store.get_envelope_by_client_id(
            "dataset-a", row.deterministic_envelope_id
        ) is None


def test_later_projection_does_not_barrier_lagging_companion_manifest(
    first_ingress_harness: FirstIngressHarness,
) -> None:
    """A companion remains authentic after a later ingress moves projection forward."""

    first = first_ingress_harness.resume_relay(row_budget=1)
    assert first.staged_rows == 1
    rows = first_ingress_harness.pending_rows()
    semantic, manifest = rows
    assert first_ingress_harness.source_row_state(semantic) == "acknowledged"
    assert first_ingress_harness.source_row_state(manifest) == "pending"
    update_cursor, updated = first_ingress_harness.push_omitted_revision_update()
    update = first_ingress_harness.store.get_envelope_by_server_cursor(update_cursor)
    state = first_ingress_harness.store.get_object_state(
        "dataset-a", "personal_context.record", updated.record_id
    )
    assert update is not None
    assert update.object_revision is None
    assert state is not None
    assert state.object_revision == 3
    assert state.latest_server_cursor == update_cursor

    result = first_ingress_harness.resume_relay(row_budget=100)

    assert result.continuation == "complete"
    assert first_ingress_harness.source_row_state(manifest) == "acknowledged"
    authority = first_ingress_harness.store.get_current_head(
        "dataset-a", "personal_context.record", updated.record_id
    )
    assert authority is not None
    assert authority.base_object_revision == 3
    assert authority.object_revision == 4


def test_authority_projection_advances_only_on_authenticated_finalize_and_retries(
    first_ingress_harness: FirstIngressHarness,
) -> None:
    """Stage stays invisible; acknowledged finalize projects exact authority once."""

    with first_ingress_harness.publications.profile_lease(
        first_ingress_harness.manifest.profile_id
    ) as lease:
        assert lease is not None
        batch = first_ingress_harness.publications.earliest_nonterminal_batch(
            first_ingress_harness.manifest.profile_id,
            row_limit=100,
        )
        assert batch is not None
        source = next(row for row in batch.rows if row.role == "semantic")
        row = replace(source, relay_owner_token=lease.owner_token)
        receipt = first_ingress_harness.service.stage_personal_context_authority(
            row, "dataset-a", "user-a"
        )
        staged_state = first_ingress_harness.store.get_object_state(
            "dataset-a", "personal_context.record", first_ingress_harness.record.record_id
        )
        assert staged_state is not None
        assert staged_state.object_revision == 1
        assert staged_state.latest_server_cursor == first_ingress_harness.ingress_cursor

        first_ingress_harness.publications.record_staged_row(
            row, server_cursor=receipt.server_cursor, lease=lease
        )
        first_ingress_harness.publications.acknowledge_row(
            row, server_cursor=receipt.server_cursor, lease=lease
        )
        first_ingress_harness.service.finalize_personal_context_authority(
            row, receipt, "dataset-a", "user-a"
        )
        projected = first_ingress_harness.store.get_object_state(
            "dataset-a", "personal_context.record", first_ingress_harness.record.record_id
        )
        assert projected is not None
        assert projected.object_revision == 2
        assert projected.object_hash == first_ingress_harness.ingress.payload_hash
        assert projected.latest_server_cursor == receipt.server_cursor
        assert projected.deleted is False

        first_ingress_harness.service.finalize_personal_context_authority(
            row, receipt, "dataset-a", "user-a"
        )
        assert (
            first_ingress_harness.store.get_object_state(
                "dataset-a",
                "personal_context.record",
                first_ingress_harness.record.record_id,
            )
            == projected
        )


def test_authority_finalize_failure_rolls_back_projection_and_apply(
    first_ingress_harness: FirstIngressHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sync authority apply and projection share one rollback boundary."""

    original = SyncDatabase.upsert_object_state
    with first_ingress_harness.publications.profile_lease(
        first_ingress_harness.manifest.profile_id
    ) as lease:
        assert lease is not None
        batch = first_ingress_harness.publications.earliest_nonterminal_batch(
            first_ingress_harness.manifest.profile_id,
            row_limit=100,
        )
        assert batch is not None
        row = replace(batch.rows[0], relay_owner_token=lease.owner_token)
        receipt = first_ingress_harness.service.stage_personal_context_authority(
            row, "dataset-a", "user-a"
        )
        first_ingress_harness.publications.record_staged_row(
            row, server_cursor=receipt.server_cursor, lease=lease
        )
        first_ingress_harness.publications.acknowledge_row(
            row, server_cursor=receipt.server_cursor, lease=lease
        )

        def fail_projection(database: SyncDatabase, *args: object, **kwargs: object):
            if database is first_ingress_harness.store.db:
                raise RuntimeError("injected projection failure")
            return original(database, *args, **kwargs)

        monkeypatch.setattr(SyncDatabase, "upsert_object_state", fail_projection)
        with pytest.raises(RuntimeError, match="injected projection failure"):
            first_ingress_harness.service.finalize_personal_context_authority(
                row, receipt, "dataset-a", "user-a"
            )

    stored = first_ingress_harness.store.get_envelope_by_server_cursor(
        receipt.server_cursor
    )
    state = first_ingress_harness.store.get_object_state(
        "dataset-a", "personal_context.record", first_ingress_harness.record.record_id
    )
    assert stored is not None
    assert stored.apply_status == "pending"
    assert state is not None
    assert state.object_revision == 1


def test_authority_finalize_rejects_a_concurrently_replaced_current_head(
    first_ingress_harness: FirstIngressHarness,
) -> None:
    """A stale staged row cannot project after its exact current head changes."""

    with first_ingress_harness.publications.profile_lease(
        first_ingress_harness.manifest.profile_id
    ) as lease:
        assert lease is not None
        batch = first_ingress_harness.publications.earliest_nonterminal_batch(
            first_ingress_harness.manifest.profile_id,
            row_limit=100,
        )
        assert batch is not None
        row = replace(batch.rows[0], relay_owner_token=lease.owner_token)
        receipt = first_ingress_harness.service.stage_personal_context_authority(
            row, "dataset-a", "user-a"
        )
        first_ingress_harness.publications.record_staged_row(
            row, server_cursor=receipt.server_cursor, lease=lease
        )
        first_ingress_harness.publications.acknowledge_row(
            row, server_cursor=receipt.server_cursor, lease=lease
        )
        with first_ingress_harness.store.db.backend.transaction() as connection:
            first_ingress_harness.store.db.execute(
                """UPDATE sync_current_heads SET latest_server_cursor = ?
                   WHERE dataset_id = ? AND domain = ? AND object_id = ?""",
                (
                    first_ingress_harness.ingress_cursor,
                    "dataset-a",
                    "personal_context.record",
                    first_ingress_harness.record.record_id,
                ),
                connection=connection,
            )

        with pytest.raises(SyncStoreError, match="authority_finalize_raced"):
            first_ingress_harness.service.finalize_personal_context_authority(
                row, receipt, "dataset-a", "user-a"
            )

    stored = first_ingress_harness.store.get_envelope_by_server_cursor(
        receipt.server_cursor
    )
    state = first_ingress_harness.store.get_object_state(
        "dataset-a", "personal_context.record", first_ingress_harness.record.record_id
    )
    assert stored is not None
    assert stored.apply_status == "pending"
    assert state is not None
    assert state.object_revision == 1


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
    assert authority_harness.source_row_state(row) == "acknowledged"
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
    assert ingress_harness.source_row_state(row) == "acknowledged"
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


def test_manifest_prestage_rejects_missing_declared_origin(
    ingress_harness: IngressHarness,
) -> None:
    """A multi-row manifest cannot become direct when its origin disappears."""

    semantic_row, _semantic_cursor = ingress_harness.persist_staged_row()
    ingress_harness.resume_relay()
    assert ingress_harness.source_row_state(semantic_row) == "acknowledged"

    with ingress_harness.claimed_row() as manifest_row:
        assert manifest_row.role == "manifest"
        assert manifest_row.batch_size == 2
        assert manifest_row.batch_ordinal == 1
    with ingress_harness.personal_db.transaction(immediate=True) as connection:
        connection.execute(
            """DELETE FROM personal_context_publication_rows
               WHERE profile_id = ? AND profile_publication_sequence = ?
                 AND batch_ordinal = ?""",
            (
                semantic_row.profile_id,
                semantic_row.profile_publication_sequence,
                semantic_row.batch_ordinal,
            ),
        )

    result = ingress_harness.resume_relay(row_budget=2)

    stored = ingress_harness.store.get_envelope_by_client_id(
        "dataset-a",
        manifest_row.deterministic_envelope_id,
    )
    assert (
        result.continuation,
        result.staged_rows,
        None if stored is None else stored.apply_status,
        ingress_harness.source_row_state(manifest_row),
        ingress_harness.has_attention(manifest_row),
    ) == (
        "personal_context_relay_pending",
        0,
        None,
        "pending",
        False,
    )


def test_direct_single_row_manifest_relays_without_origin_receipts(
    authority_harness: AuthorityHarness,
) -> None:
    """A genuine one-row manifest publication remains receipt-free."""

    relay = PersonalContextRelay(
        publications=authority_harness.publications,
        stage_authority=authority_harness.service.stage_personal_context_authority,
        finalize_authority=authority_harness.service.finalize_personal_context_authority,
        cancel_authority=authority_harness.service.cancel_personal_context_authority,
    )
    initial = relay.relay_profile(
        user_id="user-a",
        profile_id=authority_harness.manifest.profile_id,
        dataset_id="dataset-a",
        after_server_cursor=None,
        wall_time_ms=5_000,
    )
    assert initial.continuation == "complete"

    manifest_v1 = authority_harness.canonical._repository.get_manifest(
        authority_harness.manifest.profile_id
    )
    assert manifest_v1 is not None
    manifest_v2 = type(manifest_v1).model_validate(
        {
            **manifest_v1.model_dump(mode="python"),
            "revision": manifest_v1.revision + 1,
            "updated_at": manifest_v1.updated_at + timedelta(seconds=1),
            "current_version_id": "direct-manifest-v2",
        }
    )
    authority_harness.canonical._repository.commit_manifest_version(
        manifest_v2,
        expected_version_id=manifest_v1.current_version_id,
    )
    batch = authority_harness.publications.earliest_nonterminal_batch(
        manifest_v1.profile_id,
        row_limit=100,
    )
    assert batch is not None
    assert [(row.role, row.batch_ordinal, row.batch_size) for row in batch.rows] == [
        ("manifest", 0, 1)
    ]

    result = relay.relay_profile(
        user_id="user-a",
        profile_id=manifest_v1.profile_id,
        dataset_id="dataset-a",
        after_server_cursor=None,
        wall_time_ms=5_000,
    )

    assert result.continuation == "complete"
    assert authority_harness.source_row_state(batch.rows[0]) == "acknowledged"
    manifest_head = authority_harness.store.get_current_head(
        "dataset-a",
        "personal_context.manifest",
        manifest_v1.profile_id,
    )
    assert manifest_head is not None
    assert manifest_head.entity_version == manifest_v2.current_version_id
    assert manifest_head.authority is not None
    assert manifest_head.authority.role == "home_authority"
    with authority_harness.personal_db.transaction() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM personal_context_ingress_receipts"
            ).fetchone()[0]
            == 0
        )
    with authority_harness.store.db.backend.transaction() as connection:
        assert (
            authority_harness.store.db.execute(
                "SELECT COUNT(*) AS receipt_count "
                "FROM sync_personal_context_ingress_receipts",
                connection=connection,
            ).rows[0]["receipt_count"]
            == 0
        )


def test_direct_record_update_relays_manifest_without_ingress_receipts(
    authority_harness: AuthorityHarness,
) -> None:
    """A direct update may build on an applied home-authority head."""

    record_v1 = authority_harness.canonical.create_manual_record(
        scope_id=authority_harness.canonical.list_scopes()[0].scope_id,
        payload={
            "kind": "preference",
            "subject": "response.detail",
            "polarity": "like",
            "value": "concise",
        },
        semantic_key={"namespace": "preference", "subject": "response.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    relay = PersonalContextRelay(
        publications=authority_harness.publications,
        stage_authority=authority_harness.service.stage_personal_context_authority,
        finalize_authority=authority_harness.service.finalize_personal_context_authority,
        cancel_authority=authority_harness.service.cancel_personal_context_authority,
    )
    for _ in range(10):
        result = relay.relay_profile(
            user_id="user-a",
            profile_id=authority_harness.manifest.profile_id,
            dataset_id="dataset-a",
            after_server_cursor=None,
            wall_time_ms=5_000,
        )
        if result.continuation == "complete":
            break
    assert result.continuation == "complete"
    record_v1_head = authority_harness.store.get_current_head(
        "dataset-a", "personal_context.record", record_v1.record_id
    )
    manifest_v1_head = authority_harness.store.get_current_head(
        "dataset-a",
        "personal_context.manifest",
        authority_harness.manifest.profile_id,
    )
    assert record_v1_head is not None
    assert record_v1_head.authority is not None
    assert record_v1_head.authority.role == "home_authority"
    assert manifest_v1_head is not None
    assert manifest_v1_head.authority is not None
    assert manifest_v1_head.authority.role == "home_authority"

    record_v2 = authority_harness.canonical.update_record(
        record_v1.record_id,
        RecordMutation(
            payload={
                "kind": "preference",
                "subject": "response.detail",
                "polarity": "like",
                "value": "structured",
            }
        ),
        expected_version_id=record_v1.version_id,
    )
    batch_v2 = authority_harness.publications.earliest_nonterminal_batch(
        authority_harness.manifest.profile_id,
        row_limit=100,
    )
    assert batch_v2 is not None
    assert [row.role for row in batch_v2.rows] == ["semantic", "manifest"]

    for _ in range(10):
        result = relay.relay_profile(
            user_id="user-a",
            profile_id=authority_harness.manifest.profile_id,
            dataset_id="dataset-a",
            after_server_cursor=None,
            wall_time_ms=5_000,
        )
        if result.continuation == "complete":
            break

    assert result.continuation == "complete"
    assert all(
        authority_harness.source_row_state(row) == "acknowledged"
        for row in batch_v2.rows
    )
    record_v2_head = authority_harness.store.get_current_head(
        "dataset-a", "personal_context.record", record_v2.record_id
    )
    manifest_v2_head = authority_harness.store.get_current_head(
        "dataset-a",
        "personal_context.manifest",
        authority_harness.manifest.profile_id,
    )
    assert record_v2_head is not None
    assert record_v2_head.entity_version == record_v2.version_id
    assert record_v2_head.authority is not None
    assert record_v2_head.authority.role == "home_authority"
    assert manifest_v2_head is not None
    assert manifest_v2_head.server_cursor != manifest_v1_head.server_cursor
    assert manifest_v2_head.authority is not None
    assert manifest_v2_head.authority.role == "home_authority"
    with authority_harness.personal_db.transaction() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM personal_context_ingress_receipts"
            ).fetchone()[0]
            == 0
        )
    with authority_harness.store.db.backend.transaction() as connection:
        assert (
            authority_harness.store.db.execute(
                "SELECT COUNT(*) AS receipt_count "
                "FROM sync_personal_context_ingress_receipts",
                connection=connection,
            ).rows[0]["receipt_count"]
            == 0
        )
