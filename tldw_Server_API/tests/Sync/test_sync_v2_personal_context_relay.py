"""Regression coverage for ordered Personal Context publication relay."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    PersonalContextPublicationRelayStore,
    PublicationSourceBatch,
    PublicationSourceRow,
)


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
    from tldw_Server_API.app.core.Sync.v2.models import (
        PERSONAL_CONTEXT_SYNC_DOMAINS,
        SyncDatasetCreate,
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
    canonical.create_manual_record(
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
        personal_context_service_resolver=lambda _user_id: canonical,
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
                }
            },
        )
    )
    staging_errors: list[str] = []

    def stage(row, dataset_id: str, user_id: str) -> int:
        try:
            return service.stage_personal_context_authority(row, dataset_id, user_id)
        except Exception as exc:
            staging_errors.append(f"{row.domain}:{row.role}:{exc!r}")
            raise

    publications = PersonalContextPublicationRelayStore(personal_db)
    with publications.profile_lease(manifest.profile_id) as binding_lease:
        assert binding_lease is not None
        source_batch = publications.earliest_nonterminal_batch(manifest.profile_id)
        assert source_batch is not None
        source_row = replace(
            source_batch.rows[0], relay_owner_token=binding_lease.owner_token
        )
        invalid_bindings = (
            (
                replace(
                    source_row,
                    deterministic_envelope_id="forged-integrity-envelope",
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
                PersonalContextAuthoritySourceError,
            ),
            (
                replace(
                    source_row,
                    deterministic_envelope_id="wrong-generation-envelope",
                    purge_generation=1,
                ),
                "dataset-a",
                PersonalContextAuthoritySourceError,
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
            wrong_key_row = replace(
                source_row,
                deterministic_envelope_id=f"wrong-key-{key_result[0]}",
            )
            with pytest.raises(error_type):
                wrong_key_service.stage_personal_context_authority(
                    wrong_key_row, "dataset-a", "user-a"
                )
            assert store.get_envelope_by_client_id(
                "dataset-a", wrong_key_row.deterministic_envelope_id
            ) is None
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
        finalize_authority=service.finalize_personal_context_authority,
    )

    with pytest.raises(RuntimeError, match="crash after durable Sync insert"):
        relay.relay_profile(
            user_id="user-a",
            profile_id=manifest.profile_id,
            dataset_id="dataset-a",
            after_server_cursor=None,
        )
    scope_head = store.get_current_head(
        "dataset-a", "personal_context.scope", canonical.list_scopes()[0].scope_id
    )
    assert scope_head is not None
    assert scope_head.apply_status == "pending"

    publications = PersonalContextPublicationRelayStore(personal_db)
    relay = PersonalContextRelay(
        publications=publications,
        stage_authority=stage,
        finalize_authority=service.finalize_personal_context_authority,
    )
    result = relay.relay_profile(
        user_id="user-a",
        profile_id=manifest.profile_id,
        dataset_id="dataset-a",
        after_server_cursor=None,
    )

    assert result.continuation == "complete", staging_errors
    manifest_head = store.get_current_head(
        "dataset-a", "personal_context.manifest", manifest.profile_id
    )
    assert manifest_head is not None
    assert manifest_head.object_revision == 2
    assert manifest_head.base_server_cursor is not None


def test_relay_exports_a_result_type() -> None:
    """The relay boundary is available independently of HTTP transport."""

    from tldw_Server_API.app.core.Sync.v2.personal_context_relay import (
        PersonalContextRelayResult,
    )

    assert PersonalContextRelayResult.__name__ == "PersonalContextRelayResult"


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

    def earliest_nonterminal_batch(self, _profile_id: str) -> PublicationSourceBatch:
        if all(row.row_state == "acknowledged" for row in self.rows):
            return None  # type: ignore[return-value]
        return PublicationSourceBatch("profile-a", 1, "batch-a", tuple(self.rows))

    def acknowledge_row(self, row: PublicationSourceRow, *, server_cursor: int, lease: object) -> None:
        del lease
        if self.failed_after is not None and row.batch_ordinal == self.failed_after:
            raise RuntimeError("injected interruption")
        self.acknowledged.append(server_cursor)
        self.rows[row.batch_ordinal] = replace(row, row_state="acknowledged", sync_server_cursor=server_cursor)

    def complete_if_acknowledged(
        self, _batch: PublicationSourceBatch, *, lease: object
    ) -> bool:
        assert lease.owner_token == "lease-token"
        return all(row.row_state == "acknowledged" for row in self.rows)

    def mark_attention(self, _batch: PublicationSourceBatch) -> None:
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


def test_relay_never_stages_manifest_before_semantic_siblings() -> None:
    from tldw_Server_API.app.core.Sync.v2.personal_context_relay import PersonalContextRelay

    publications = _Publications()
    staged: list[str] = []
    relay = PersonalContextRelay(
        publications=publications,
        stage_authority=lambda row, _dataset, _user: staged.append(row.role) or row.batch_ordinal + 1,
    )

    publications.failed_after = 0
    with pytest.raises(RuntimeError, match="injected interruption"):
        relay.relay_profile(user_id="user-a", profile_id="profile-a", dataset_id="dataset-a", after_server_cursor=None)
    assert staged == ["semantic"]
    publications.failed_after = None
    relay.relay_profile(user_id="user-a", profile_id="profile-a", dataset_id="dataset-a", after_server_cursor=None)
    assert staged == ["semantic", "semantic", "manifest"]


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
