"""Exact ingress receipts permit retryable repair without reviving terminal rows."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Personalization.personal_context_publication import CanonicalApplyReceipt
from tldw_Server_API.app.core.Sync.v2.models import PERSONAL_CONTEXT_SYNC_DOMAINS, SyncDatasetCreate, SyncDeviceUpsert
from tldw_Server_API.app.core.Sync.v2.store import SyncStoreError, SyncV2Store

pytestmark = pytest.mark.unit


@pytest.fixture(params=["sqlite", "postgres"])
def ingress_store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[tuple[SyncV2Store, int, CanonicalApplyReceipt]]:
    """Use real SQLite and the shared isolated PostgreSQL database fixture."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        database = SyncDatabase(backend=backend)
    else:
        database = SyncDatabase(sqlite_path=tmp_path / "ingress-repair.db")
    store = SyncV2Store(database)
    store.upsert_device(
        SyncDeviceUpsert(
            device_id="repair-device",
            user_id="repair-user",
            display_name="Repair fixture",
            client_type="chatbook",
        )
    )
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="repair-dataset",
            owner_user_id="repair-user",
            encryption_policy="server_trusted_v1",
            domains=sorted(PERSONAL_CONTEXT_SYNC_DOMAINS),
        )
    )
    # Seed this narrow storage fixture directly: ordinary PostgreSQL insertion
    # currently hits an unrelated _ensure_domain_state placeholder mismatch.
    # The bootstrap regression covers the complete SQLite push/repair workflow.
    with database.backend.transaction() as connection:
        database.execute(
            """INSERT INTO sync_envelopes (
               server_sequence, dataset_id, domain, entity_id, operation,
               client_envelope_id, device_id, server_timestamp, entity_version,
               routing_metadata_json, payload_ciphertext, payload_hash,
               payload_size_bytes, adapter_version, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                1,
                "repair-dataset",
                "personal_context.record",
                "repair-record",
                "upsert",
                "repair-client-envelope",
                "repair-device",
                "2026-09-04T00:00:00Z",
                '"record-v1"',
                '{"personal_context_authority":{"role":"client_ingress"}}',
                "opaque-fixture-ciphertext",
                "hmac-sha256-v1:" + "a" * 64,
                24,
                1,
                "accepted",
            ),
            connection=connection,
        )
    receipt = CanonicalApplyReceipt(
        resulting_object_id="repair-record",
        resulting_version_id="record-v1",
        manifest_revision=1,
        manifest_version_id="manifest-v1",
        purge_generation=0,
        publication_batch_id="repair-publication",
        profile_publication_sequence=1,
        receipt_id="repair-exact-receipt",
        dataset_id="repair-dataset",
        device_id="repair-device",
        client_envelope_id="repair-client-envelope",
        canonical_payload_digest="sha256:" + "a" * 64,
        wire_entity_version="record-v1",
    )
    try:
        yield store, 1, receipt
    finally:
        if backend is not None:
            backend.get_pool().close_all()


def _set_apply_state(store: SyncV2Store, cursor: int, state: str) -> None:
    """Set only the retry state and sanitized failure fields under a real transaction."""
    with store.db.backend.transaction() as connection:
        store.db.execute(
            "UPDATE sync_envelopes SET apply_status = ?, apply_error_code = ?, apply_error_message = ? WHERE server_sequence = ?",
            (state, "fixture_failure", "fixture failure", cursor),
            connection=connection,
        )


@pytest.mark.parametrize("state", ["pending", "failed", "applied"])
def test_verified_receipt_repairs_only_retryable_ingress(
    ingress_store: tuple[SyncV2Store, int, CanonicalApplyReceipt],
    state: str,
) -> None:
    """Exact failed repair commits once and clears old sanitized error fields."""
    store, cursor, receipt = ingress_store
    _set_apply_state(store, cursor, state)
    applied = store.mark_personal_context_ingress_applied(server_cursor=cursor, receipt=receipt)
    assert applied.apply_status == "applied"
    assert applied.apply_error_code is None
    assert applied.apply_error_message is None
    stored_receipt = store.get_personal_context_ingress_receipt(cursor)
    replay = store.mark_personal_context_ingress_applied(server_cursor=cursor, receipt=receipt)
    assert replay.apply_status == "applied"
    assert store.get_personal_context_ingress_receipt(cursor) == stored_receipt


@pytest.mark.parametrize("state", ["conflict", "superseded", "skipped"])
def test_verified_receipt_cannot_revive_nonretryable_ingress(
    ingress_store: tuple[SyncV2Store, int, CanonicalApplyReceipt],
    state: str,
) -> None:
    """Receipt verification never authorizes reviving conflict or terminal states."""
    store, cursor, receipt = ingress_store
    _set_apply_state(store, cursor, state)
    with pytest.raises(SyncStoreError, match="personal_context_ingress_receipt_mismatch"):
        store.mark_personal_context_ingress_applied(server_cursor=cursor, receipt=receipt)
    assert store.get_envelope_by_server_cursor(cursor).apply_status == state
    assert store.get_personal_context_ingress_receipt(cursor) is None


@pytest.mark.parametrize(
    "field", ["dataset_id", "device_id", "wire_entity_version", "canonical_payload_digest", "receipt_id"]
)
def test_failed_ingress_repair_rejects_changed_durable_receipt(
    ingress_store: tuple[SyncV2Store, int, CanonicalApplyReceipt],
    field: str,
) -> None:
    """No mismatched identity or receipt can repair a retryable failed row."""
    store, cursor, receipt = ingress_store
    store.mark_personal_context_ingress_applied(server_cursor=cursor, receipt=receipt)
    original = store.get_personal_context_ingress_receipt(cursor)
    _set_apply_state(store, cursor, "failed")
    with pytest.raises(SyncStoreError, match="personal_context_ingress_receipt_mismatch"):
        store.mark_personal_context_ingress_applied(
            server_cursor=cursor, receipt=replace(receipt, **{field: "changed-value"})
        )
    assert store.get_envelope_by_server_cursor(cursor).apply_status == "failed"
    assert store.get_personal_context_ingress_receipt(cursor) == original
