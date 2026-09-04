"""Crash recovery and CAS regressions for Personal Context authority relay."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.Personalization_DB import (
    PersonalizationDB,
)
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    PersonalContextPublicationJournal,
    PersonalContextPublicationRelayStore,
    PublicationRelayLease,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository import (
    PersonalContextRepository,
)
from tldw_Server_API.app.core.Personalization.personal_context_service import (
    PersonalContextService,
)
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AdapterConflict,
    AdapterRejected,
    SyncAdapterRegistry,
)
from tldw_Server_API.app.core.Sync.v2.domain_adapters.personal_context import (
    PersonalContextDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.models import PERSONAL_CONTEXT_SYNC_DOMAINS
from tldw_Server_API.app.core.Sync.v2.personal_context_relay import (
    PersonalContextRelay,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store
from tldw_Server_API.tests.Sync.test_sync_v2_personal_context_authority_identity import (
    AuthorityHarness,
)


@pytest.fixture
def authority_harness(tmp_path, monkeypatch) -> AuthorityHarness:
    """Create the existing real two-database authority harness."""

    return AuthorityHarness(tmp_path, monkeypatch)


def _restart(harness: AuthorityHarness) -> SimpleNamespace:
    """Rebuild every relay-facing object over the same two persisted files."""

    personal_db = PersonalizationDB.for_path(harness.personal_db.db_path)
    canonical = PersonalContextService(PersonalContextRepository(personal_db))
    _key_id, integrity_key = canonical.sync_integrity_key(harness.manifest.profile_id)
    sync_path = Path(harness.personal_db.db_path).with_name("sync.db")
    store = SyncV2Store(SyncDatabase(sqlite_path=sync_path))
    service = SyncV2Service(
        store=store,
        adapters=SyncAdapterRegistry(
            [
                PersonalContextDomainAdapter(
                    domain=domain,
                    integrity_key_resolver=lambda _dataset, _requested: integrity_key,
                    encryption_key_resolver=lambda _dataset: (b"e" * 32, 1),
                )
                for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
            ]
        ),
        personal_context_service_resolver=lambda _user_id: canonical,
    )
    publications = PersonalContextPublicationRelayStore(personal_db)
    return SimpleNamespace(
        personal_db=personal_db,
        canonical=canonical,
        store=store,
        service=service,
        publications=publications,
    )


def _relay(runtime: Any, *, publications: Any | None = None, row_budget: int = 100):
    return PersonalContextRelay(
        publications=publications or runtime.publications,
        stage_authority=runtime.service.stage_personal_context_authority,
        finalize_authority=runtime.service.finalize_personal_context_authority,
        cancel_authority=runtime.service.cancel_personal_context_authority,
    ).relay_profile(
        user_id="user-a",
        profile_id=runtime.canonical.get_manifest().profile_id,
        dataset_id="dataset-a",
        after_server_cursor=None,
        row_budget=row_budget,
        wall_time_ms=5_000,
    )


@contextmanager
def _source_row(runtime: Any):
    with runtime.publications.profile_lease(
        runtime.canonical.get_manifest().profile_id
    ) as lease:
        assert lease is not None
        batch = runtime.publications.earliest_nonterminal_batch(
            runtime.canonical.get_manifest().profile_id,
            row_limit=100,
        )
        assert batch is not None
        yield replace(batch.rows[0], relay_owner_token=lease.owner_token), lease


def _stored_source_state(runtime: Any, row: Any) -> tuple[str, int | None]:
    with runtime.personal_db.transaction() as connection:
        stored = connection.execute(
            """SELECT row_state, sync_server_cursor
               FROM personal_context_publication_rows
               WHERE profile_id = ? AND profile_publication_sequence = ?
                 AND batch_ordinal = ?""",
            (row.profile_id, row.profile_publication_sequence, row.batch_ordinal),
        ).fetchone()
    assert stored is not None
    return str(stored["row_state"]), stored["sync_server_cursor"]


def _drain(runtime: Any) -> Any:
    """Relay bounded batches until the persisted source is exhausted."""

    result = None
    for _attempt in range(10):
        result = _relay(runtime)
        if result.continuation == "complete":
            return result
    pytest.fail(f"relay did not converge: {result}")


@pytest.mark.parametrize(
    "boundary",
    ["insert", "source_staged", "source_ack", "finalize", "complete"],
)
def test_restart_after_each_durable_boundary_converges_exactly_once(
    authority_harness: AuthorityHarness,
    boundary: str,
) -> None:
    """Every one-DB commit boundary is replayable after full reconstruction."""

    publications = authority_harness.publications
    original_record = publications.record_staged_row
    original_ack = publications.acknowledge_row
    original_complete = publications.complete_if_acknowledged
    captured: dict[str, Any] = {}
    failed = False

    def fail_once(name: str) -> None:
        nonlocal failed
        if boundary == name and not failed:
            failed = True
            raise RuntimeError(f"injected after {name}")

    def stage(row, dataset_id, user_id):
        receipt = authority_harness.service.stage_personal_context_authority(
            row, dataset_id, user_id
        )
        captured.setdefault("row", row)
        captured.setdefault("receipt", receipt)
        fail_once("insert")
        return receipt

    def record(row, *, server_cursor, lease):
        original_record(row, server_cursor=server_cursor, lease=lease)
        fail_once("source_staged")

    def acknowledge(row, *, server_cursor, lease):
        original_ack(row, server_cursor=server_cursor, lease=lease)
        fail_once("source_ack")

    def finalize(row, receipt, dataset_id, user_id):
        authority_harness.service.finalize_personal_context_authority(
            row, receipt, dataset_id, user_id
        )
        fail_once("finalize")

    def complete(batch, *, lease):
        completed = original_complete(batch, lease=lease)
        if completed:
            fail_once("complete")
        return completed

    publications.record_staged_row = record
    publications.acknowledge_row = acknowledge
    publications.complete_if_acknowledged = complete
    first = PersonalContextRelay(
        publications=publications,
        stage_authority=stage,
        finalize_authority=finalize,
        cancel_authority=authority_harness.service.cancel_personal_context_authority,
    ).relay_profile(
        user_id="user-a",
        profile_id=authority_harness.manifest.profile_id,
        dataset_id="dataset-a",
        after_server_cursor=None,
        wall_time_ms=5_000,
    )

    assert failed is True
    assert first.continuation == "personal_context_relay_pending"
    receipt = captured["receipt"]
    interrupted = authority_harness.store.get_envelope_by_server_cursor(
        receipt.server_cursor
    )
    assert interrupted is not None
    assert interrupted.apply_status == (
        "applied" if boundary in {"finalize", "complete"} else "pending"
    )

    restarted = _restart(authority_harness)
    recovered = _drain(restarted)
    stored = restarted.store.get_envelope_by_server_cursor(receipt.server_cursor)
    duplicate = restarted.store.get_envelope_by_client_id(
        "dataset-a", receipt.deterministic_envelope_id
    )

    assert recovered.continuation == "complete"
    assert stored is not None and stored.apply_status == "applied"
    assert duplicate is not None and duplicate.server_cursor == receipt.server_cursor
    assert _stored_source_state(restarted, captured["row"]) == (
        "acknowledged",
        receipt.server_cursor,
    )


def test_authority_is_hidden_until_source_acknowledgement(
    authority_harness: AuthorityHarness,
) -> None:
    """Moving finalization before source acknowledgement must fail this test."""

    original = authority_harness.publications.acknowledge_row

    def acknowledge_only_while_hidden(row, *, server_cursor, lease) -> None:
        stored = authority_harness.store.get_envelope_by_server_cursor(server_cursor)
        assert stored is not None
        assert stored.apply_status == "pending"
        original(row, server_cursor=server_cursor, lease=lease)

    authority_harness.publications.acknowledge_row = acknowledge_only_while_hidden
    result = _relay(authority_harness)

    assert result.continuation in {"complete", "personal_context_relay_pending"}


def test_restart_after_source_staged_commit_keeps_hidden_row_for_recovery(
    authority_harness: AuthorityHarness,
) -> None:
    """A post-commit exception must not cancel a cursor already stored by source."""

    original = authority_harness.publications.record_staged_row
    captured: dict[str, Any] = {}

    def fail_after_record(row, *, server_cursor, lease) -> None:
        original(row, server_cursor=server_cursor, lease=lease)
        captured.update(row=row, cursor=server_cursor)
        raise RuntimeError("injected after source staged commit")

    authority_harness.publications.record_staged_row = fail_after_record
    first = _relay(authority_harness)

    assert first.continuation == "personal_context_relay_pending"
    assert _stored_source_state(authority_harness, captured["row"]) == (
        "staged",
        captured["cursor"],
    )
    staged = authority_harness.store.get_envelope_by_server_cursor(captured["cursor"])
    assert staged is not None
    assert staged.apply_status == "pending"

    restarted = _restart(authority_harness)
    recovered = _relay(restarted, row_budget=1)
    stored = restarted.store.get_envelope_by_server_cursor(captured["cursor"])

    assert recovered.staged_rows == 1
    assert stored is not None
    assert stored.apply_status == "applied"
    assert _stored_source_state(restarted, captured["row"]) == (
        "acknowledged",
        captured["cursor"],
    )


def test_purge_after_orphan_insert_compensates_only_hidden_authority(
    authority_harness: AuthorityHarness,
) -> None:
    """A terminalized deterministic source still identifies its hidden orphan."""

    _drain(authority_harness)
    applied = authority_harness.store.get_current_head(
        "dataset-a",
        "personal_context.scope",
        authority_harness.canonical.list_scopes()[0].scope_id,
    )
    assert applied is not None and applied.apply_status == "applied"
    authority_harness.canonical.create_manual_record(
        scope_id=authority_harness.canonical.list_scopes()[0].scope_id,
        payload={
            "kind": "preference",
            "subject": "relay.recovery",
            "polarity": "like",
            "value": "deterministic",
        },
        semantic_key={"namespace": "preference", "subject": "relay.recovery"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    captured: dict[str, Any] = {}

    def fail_after_insert(row, dataset_id, user_id):
        receipt = authority_harness.service.stage_personal_context_authority(
            row, dataset_id, user_id
        )
        captured.update(row=row, receipt=receipt)
        raise RuntimeError("injected after authority insert")

    relay = PersonalContextRelay(
        publications=authority_harness.publications,
        stage_authority=fail_after_insert,
        finalize_authority=authority_harness.service.finalize_personal_context_authority,
        cancel_authority=authority_harness.service.cancel_personal_context_authority,
    )
    first = relay.relay_profile(
        user_id="user-a",
        profile_id=authority_harness.manifest.profile_id,
        dataset_id="dataset-a",
        after_server_cursor=None,
        wall_time_ms=5_000,
    )
    assert first.continuation == "personal_context_relay_pending"

    authority_harness.canonical.purge_profile(
        mode="everywhere",
        confirmation="DELETE EVERYWHERE",
        expected_purge_generation=0,
    )
    restarted = _restart(authority_harness)
    result = _relay(restarted, row_budget=1)

    assert result.continuation == "personal_context_relay_pending"
    assert (
        restarted.store.get_envelope_by_server_cursor(
            captured["receipt"].server_cursor
        )
        is None
    )
    retained = restarted.store.get_envelope_by_server_cursor(applied.server_cursor)
    assert retained is not None and retained.apply_status == "applied"


def test_stale_batch_state_cannot_record_a_stage_receipt(
    authority_harness: AuthorityHarness,
) -> None:
    """A stale source snapshot cannot cross a terminal batch CAS."""

    with _source_row(authority_harness) as claimed:
        row, lease = claimed
        receipt = authority_harness.service.stage_personal_context_authority(
            row, "dataset-a", "user-a"
        )
        with authority_harness.personal_db.transaction(immediate=True) as connection:
            connection.execute(
                """UPDATE personal_context_publication_batches
                   SET status = 'purge_terminal'
                   WHERE profile_id = ? AND profile_publication_sequence = ?""",
                (row.profile_id, row.profile_publication_sequence),
            )

        with pytest.raises(RuntimeError, match="source claim changed"):
            authority_harness.publications.record_staged_row(
                row,
                server_cursor=receipt.server_cursor,
                lease=PublicationRelayLease(row.profile_id, lease.owner_token),
            )

        assert _stored_source_state(authority_harness, row) == ("pending", None)


def test_lease_release_detects_owner_race(authority_harness: AuthorityHarness) -> None:
    """A stolen or expired owner token is not reported as a successful release."""

    with pytest.raises(RuntimeError, match="lease changed"):
        with authority_harness.publications.profile_lease(
            authority_harness.manifest.profile_id
        ) as lease:
            assert lease is not None
            with authority_harness.personal_db.transaction(immediate=True) as connection:
                connection.execute(
                    """UPDATE personal_context_publication_relay_leases
                       SET owner_token = 'other-owner', expires_at_ns = 0
                       WHERE profile_id = ? AND owner_token = ?""",
                    (authority_harness.manifest.profile_id, lease.owner_token),
                )


def test_lease_release_detects_expired_owner(authority_harness: AuthorityHarness) -> None:
    """An expired lease is not released as though it remained live."""

    with pytest.raises(RuntimeError, match="lease changed"):
        with authority_harness.publications.profile_lease(
            authority_harness.manifest.profile_id
        ) as lease:
            assert lease is not None
            with authority_harness.personal_db.transaction(immediate=True) as connection:
                connection.execute(
                    """UPDATE personal_context_publication_relay_leases
                       SET expires_at_ns = 0
                       WHERE profile_id = ? AND owner_token = ?""",
                    (authority_harness.manifest.profile_id, lease.owner_token),
                )


def test_two_relay_instances_fence_expired_owner(
    authority_harness: AuthorityHarness,
) -> None:
    """Only the replacement instance can act after the first durable lease expires."""

    first = authority_harness.publications
    second = PersonalContextPublicationRelayStore(authority_harness.personal_db)
    with pytest.raises(RuntimeError, match="lease changed"):
        with first.profile_lease(authority_harness.manifest.profile_id) as first_lease:
            assert first_lease is not None
            with second.profile_lease(authority_harness.manifest.profile_id) as blocked:
                assert blocked is None
            with authority_harness.personal_db.transaction(immediate=True) as connection:
                connection.execute(
                    """UPDATE personal_context_publication_relay_leases
                       SET expires_at_ns = 0
                       WHERE profile_id = ? AND owner_token = ?""",
                    (
                        authority_harness.manifest.profile_id,
                        first_lease.owner_token,
                    ),
                )
            with second.profile_lease(authority_harness.manifest.profile_id) as replacement:
                assert replacement is not None
                assert replacement.owner_token != first_lease.owner_token


def test_corrupt_source_restart_remains_poisoned_across_both_stores(
    authority_harness: AuthorityHarness,
) -> None:
    """Authenticated source corruption remains durable after full reconstruction."""

    with authority_harness.personal_db.transaction(immediate=True) as connection:
        source = connection.execute(
            """SELECT deterministic_envelope_id
               FROM personal_context_publication_rows
               WHERE profile_id = ? AND profile_publication_sequence = 1
                 AND batch_ordinal = 0""",
            (authority_harness.manifest.profile_id,),
        ).fetchone()
        assert source is not None
        connection.execute(
            """UPDATE personal_context_publication_rows SET ciphertext = ?
               WHERE profile_id = ? AND profile_publication_sequence = 1
                 AND batch_ordinal = 0""",
            (b"corrupt", authority_harness.manifest.profile_id),
        )

    first = _relay(authority_harness)
    restarted = _restart(authority_harness)
    second = _relay(restarted)

    assert first.continuation == "relay_poisoned"
    assert second.continuation == "relay_poisoned"
    assert restarted.store.get_envelope_by_client_id(
        "dataset-a", str(source["deterministic_envelope_id"])
    ) is None
    with restarted.personal_db.transaction() as connection:
        attention = connection.execute(
            """SELECT error_code FROM personal_context_publication_relay_attention
               WHERE profile_id = ? AND profile_publication_sequence = 1""",
            (authority_harness.manifest.profile_id,),
        ).fetchone()
    assert attention is not None and attention["error_code"] == "relay_poisoned"


def test_stale_corrupt_source_cannot_persist_poison_after_lease_takeover(
    authority_harness: AuthorityHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale decrypt failure cannot poison the replacement owner's source batch."""

    def lose_lease_then_fail(_journal, _row):
        with authority_harness.personal_db.transaction(immediate=True) as connection:
            connection.execute(
                """UPDATE personal_context_publication_relay_leases
                   SET owner_token = 'replacement-owner',
                       expires_at_ns = 9223372036854775807
                   WHERE profile_id = ?""",
                (authority_harness.manifest.profile_id,),
            )
        raise ValueError("injected authenticated source failure")

    monkeypatch.setattr(
        PersonalContextPublicationJournal,
        "decrypt_row",
        lose_lease_then_fail,
    )
    result = _relay(authority_harness)

    assert result.continuation == "personal_context_relay_pending"
    with authority_harness.personal_db.transaction() as connection:
        attention = connection.execute(
            """SELECT 1 FROM personal_context_publication_relay_attention
               WHERE profile_id = ?""",
            (authority_harness.manifest.profile_id,),
        ).fetchone()
    assert attention is None


@pytest.mark.parametrize("failure_kind", ["adapter", "head", "database"])
def test_retryable_stage_failures_never_create_durable_poison(
    authority_harness: AuthorityHarness,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    """Adapter, head, and DB failures leave the authenticated source retryable."""

    with _source_row(authority_harness) as claimed:
        row, _lease = claimed
    diagnostics: list[str] = []
    original_head = authority_harness.store.get_current_head
    sink = logger.add(diagnostics.append, level="WARNING", format="{level}:{message}")
    try:
        if failure_kind == "database":
            monkeypatch.setattr(
                authority_harness.store,
                "get_current_head",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    DatabaseError("injected database failure")
                ),
            )
        elif failure_kind == "adapter":
            monkeypatch.setattr(
                authority_harness.service,
                "_evaluate_envelope",
                lambda _dataset, envelope: AdapterRejected(
                    client_envelope_id=envelope.client_envelope_id,
                    error_code="injected_adapter_rejection",
                    message="retry without source poison",
                ),
            )
        else:
            monkeypatch.setattr(
                authority_harness.service,
                "_evaluate_envelope",
                lambda _dataset, envelope: AdapterConflict(
                    client_envelope_id=envelope.client_envelope_id,
                    domain=envelope.domain,
                    entity_id=envelope.object_id,
                    conflict_type="injected_head_contention",
                ),
            )
        result = _relay(authority_harness)
    finally:
        logger.remove(sink)
        if failure_kind == "database":
            monkeypatch.setattr(
                authority_harness.store, "get_current_head", original_head
            )

    assert result.continuation == "personal_context_relay_pending"
    assert authority_harness.store.get_envelope_by_client_id(
        "dataset-a", row.deterministic_envelope_id
    ) is None
    with authority_harness.personal_db.transaction() as connection:
        attention = connection.execute(
            """SELECT 1 FROM personal_context_publication_relay_attention
               WHERE profile_id = ?""",
            (authority_harness.manifest.profile_id,),
        ).fetchone()
    assert attention is None
    assert all("poison" not in message.lower() for message in diagnostics)


def test_swallowed_failure_then_head_contention_remains_recoverable(
    authority_harness: AuthorityHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An earlier uncertain attempt cannot turn later head contention into poison."""

    original = authority_harness.service._evaluate_envelope
    attempts = 0

    def fail_then_contend(_dataset, envelope):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise DatabaseError("injected swallowed failure")
        return AdapterConflict(
            client_envelope_id=envelope.client_envelope_id,
            domain=envelope.domain,
            entity_id=envelope.object_id,
            conflict_type="injected_head_contention",
        )

    monkeypatch.setattr(
        authority_harness.service, "_evaluate_envelope", fail_then_contend
    )
    first = _relay(authority_harness)
    second = _relay(authority_harness)
    monkeypatch.setattr(authority_harness.service, "_evaluate_envelope", original)

    assert first.continuation == "personal_context_relay_pending"
    assert second.continuation == "personal_context_relay_pending"
    with authority_harness.personal_db.transaction() as connection:
        attention = connection.execute(
            """SELECT 1 FROM personal_context_publication_relay_attention
               WHERE profile_id = ?""",
            (authority_harness.manifest.profile_id,),
        ).fetchone()
    assert attention is None
    assert _drain(authority_harness).continuation == "complete"


def test_relay_artifacts_and_diagnostics_never_store_plaintext_canary(
    authority_harness: AuthorityHarness,
) -> None:
    """Both active DBs, WAL/SHM files, and captured logs remain content-free."""

    canary = "TASK-13168-private-canary-9f443de76a"
    authority_harness.canonical.create_manual_record(
        scope_id=authority_harness.canonical.list_scopes()[0].scope_id,
        payload={
            "kind": "preference",
            "subject": "relay.canary",
            "polarity": "like",
            "value": canary,
        },
        semantic_key={"namespace": "preference", "subject": "relay.canary"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    diagnostics: list[str] = []
    sink = logger.add(diagnostics.append, level="DEBUG", format="{level}:{message}")
    try:
        assert _drain(authority_harness).continuation == "complete"
    finally:
        logger.remove(sink)

    personal_path = Path(authority_harness.personal_db.db_path)
    sync_path = personal_path.with_name("sync.db")
    artifacts = [
        candidate
        for base in (personal_path, sync_path)
        for candidate in (base, Path(f"{base}-wal"), Path(f"{base}-shm"))
        if candidate.exists()
    ]

    assert personal_path in artifacts and sync_path in artifacts
    assert all(canary.encode() not in artifact.read_bytes() for artifact in artifacts)
    assert all(canary not in message for message in diagnostics)
