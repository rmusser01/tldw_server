from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.models import (
    M1_SYNC_DOMAINS,
    NOTES_ORGANIZATION_DOMAINS,
    SyncDatasetCreate,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
    _materialization_plan_hash,
)
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _metadata(state: str, bootstrap_id: str = "bootstrap-current") -> dict[str, object]:
    return {
        "notes_organization_v1": {
            "bootstrap_id": bootstrap_id,
            "state": state,
            "captured_count": 0,
            "expected_count": 0,
            "error_code": None,
        }
    }


def _store(tmp_path: Path, *, state: str) -> SyncV2Store:
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "gate.sqlite"))
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            domains=[*M1_SYNC_DOMAINS, *NOTES_ORGANIZATION_DOMAINS],
            metadata=_metadata(state),
        )
    )
    return store


def _keyword_envelope(*, object_id: str, client_id: str) -> SyncEnvelopeCreate:
    return SyncEnvelopeCreate(
        dataset_id="dataset-1",
        client_envelope_id=client_id,
        domain="notes.keyword",
        operation="upsert",
        object_id=object_id,
        object_revision=1,
        payload={"keyword": "Research"},
        payload_hash=f"sha256:{client_id}",
    )


def _group_envelope(*, object_id: str, group_id: str) -> SyncEnvelopeCreate:
    envelope = replace(
        _keyword_envelope(object_id=object_id, client_id=f"env-{group_id}"),
        device_id="__server__",
        mutation_group_id=group_id,
        mutation_step=0,
        mutation_step_count=1,
        mutation_plan_hash="0" * 64,
    )
    return replace(
        envelope,
        mutation_plan_hash=_materialization_plan_hash([envelope]),
    )


def test_single_append_rechecks_notes_organization_readiness_in_its_transaction(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, state="ready")
    stale_ready = store.get_dataset("dataset-1")
    assert stale_ready is not None
    store.transition_notes_organization_bootstrap(
        "dataset-1",
        bootstrap_id="bootstrap-current",
        expected_state="ready",
        state="failed",
        captured_count=0,
        expected_count=0,
        error_code="notes_organization_bootstrap_source_invalid",
    )

    with pytest.raises(SyncStoreError, match="notes_organization_sync_not_ready"):
        store.insert_envelope(
            _keyword_envelope(
                object_id="11111111-1111-4111-8111-111111111111",
                client_id="single-after-stale-ready",
            )
        )

    core = store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id="dataset-1",
            client_envelope_id="core-while-organization-failed",
            domain="notes.note",
            operation="upsert",
            object_id="note-core",
            object_revision=1,
            payload={"title": "Core remains available"},
            payload_hash="sha256:core-while-organization-failed",
        )
    )
    assert core.domain == "notes.note"


def test_group_append_requires_ready_or_exact_trusted_initializing_bootstrap(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, state="initializing")
    ordinary = _group_envelope(
        object_id="22222222-2222-4222-8222-222222222222",
        group_id="ordinary-group",
    )

    with pytest.raises(SyncStoreError, match="notes_organization_sync_not_ready"):
        store.insert_envelopes_atomic([ordinary])
    with pytest.raises(SyncStoreError, match="notes_organization_sync_not_ready"):
        store.insert_envelopes_atomic(
            [ordinary],
            trusted_notes_organization_bootstrap_id="bootstrap-stale",
        )

    trusted = _group_envelope(
        object_id="33333333-3333-4333-8333-333333333333",
        group_id="trusted-group",
    )
    inserted = store.insert_envelopes_atomic(
        [trusted],
        trusted_notes_organization_bootstrap_id="bootstrap-current",
    )
    assert [item.client_envelope_id for item in inserted] == [trusted.client_envelope_id]


def test_ready_cas_rejects_count_mismatch_undrained_steps_and_stale_worker(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, state="initializing")

    with pytest.raises(SyncStoreError, match="notes_organization_bootstrap_verification_failed"):
        store.transition_notes_organization_bootstrap(
            "dataset-1",
            bootstrap_id="bootstrap-current",
            expected_state="initializing",
            state="ready",
            captured_count=0,
            expected_count=1,
            ready_verifier=lambda: True,
        )

    pending = store.insert_envelopes_atomic(
        [
            _group_envelope(
                object_id="44444444-4444-4444-8444-444444444444",
                group_id="pending-bootstrap-group",
            )
        ],
        trusted_notes_organization_bootstrap_id="bootstrap-current",
    )[0]
    with pytest.raises(SyncStoreError, match="notes_organization_bootstrap_verification_failed"):
        store.transition_notes_organization_bootstrap(
            "dataset-1",
            bootstrap_id="bootstrap-current",
            expected_state="initializing",
            state="ready",
            captured_count=1,
            expected_count=1,
            ready_verifier=lambda: True,
        )

    assert pending.server_cursor is not None
    store.mark_bootstrap_envelope_verified(
        pending.server_cursor, bootstrap_id="bootstrap-current"
    )
    ready = store.transition_notes_organization_bootstrap(
        "dataset-1",
        bootstrap_id="bootstrap-current",
        expected_state="initializing",
        state="ready",
        captured_count=1,
        expected_count=1,
        ready_verifier=lambda: True,
    )
    assert ready.metadata["notes_organization_v1"]["state"] == "ready"

    with pytest.raises(SyncStoreError, match="notes_organization_bootstrap_compare_and_set_failed"):
        store.transition_notes_organization_bootstrap(
            "dataset-1",
            bootstrap_id="bootstrap-stale",
            expected_state="ready",
            state="failed",
            captured_count=1,
            expected_count=1,
            error_code="notes_organization_bootstrap_capture_failed",
        )
