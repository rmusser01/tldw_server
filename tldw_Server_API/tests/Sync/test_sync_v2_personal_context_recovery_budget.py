"""Exact row and deadline budgets for Personal Context pull recovery."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    AuthorityStageReceipt,
    PersonalContextPublicationJournal,
    PublicationSourceBatch,
    PublicationSourceRow,
)
from tldw_Server_API.app.core.Sync.v2.personal_context_relay import (
    PersonalContextRelay,
    PersonalContextRelayResult,
)
from tldw_Server_API.app.core.Sync.v2.service import (
    _PersonalContextRecoveryBudget,
)
from tldw_Server_API.tests.Sync.test_sync_v2_personal_context_authority_identity import (
    AuthorityHarness,
)
from tldw_Server_API.tests.Sync.test_sync_v2_personal_context_transport import (
    DATASET_ID,
    DOMAIN,
    EXCHANGE,
    _insert_authority,
    _insert_hidden_ingress,
    _service,
)

pytestmark = pytest.mark.unit


class _ManualClock:
    def __init__(self, now_ns: int = 0) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


class _BudgetConsumingRelay:
    """Small source-side seam that spends the service-owned budget."""

    def __init__(self, source_rows: int) -> None:
        self.source_rows = source_rows
        self.budgets: list[_PersonalContextRecoveryBudget] = []

    def relay_profile(self, **values: Any) -> PersonalContextRelayResult:
        budget = values["budget"]
        self.budgets.append(budget)
        consumed = 0
        while consumed < self.source_rows and budget.consume():
            consumed += 1
        complete = consumed == self.source_rows
        return PersonalContextRelayResult(
            staged_rows=consumed,
            source_exhausted=complete,
            visible_lookahead=False,
            continuation=("complete" if complete else "personal_context_relay_pending"),
            inspected_rows=consumed,
        )


@pytest.mark.parametrize(
    ("source_rows", "raw_rows", "expected_raw"),
    [(100, 0, 0), (0, 100, 100), (40, 60, 60), (99, 1, 1), (100, 1, 0), (99, 2, 1)],
)
def test_legacy_pull_spends_one_exact_source_plus_raw_budget(
    tmp_path,
    source_rows: int,
    raw_rows: int,
    expected_raw: int,
) -> None:
    """Resetting either phase to 100 would inspect more than 100 rows."""

    service, _target, _sqlite_path = _service(tmp_path)
    relay = _BudgetConsumingRelay(source_rows)
    service.personal_context_relay = relay
    service._recovery_clock_ns = _ManualClock()
    raw_cursors = [
        _insert_hidden_ingress(service, ordinal=index) for index in range(raw_rows)
    ]

    pulled = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=[DOMAIN],
        cursor=0,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )

    assert len(relay.budgets) == 1
    budget = relay.budgets[0]
    assert 100 - budget.remaining_rows == min(100, source_rows + raw_rows)
    expected_cursor = raw_cursors[expected_raw - 1] if expected_raw else 0
    assert pulled.next_cursor == str(expected_cursor)
    if source_rows + raw_rows > 100:
        assert pulled.has_more is True


@pytest.mark.parametrize("signed", [False, True])
def test_mixed_pull_modes_share_the_same_relay_and_scan_budget(
    tmp_path,
    signed: bool,
) -> None:
    """Signed and legacy routing must not create separate recovery allowances."""

    service, _target, _sqlite_path = _service(tmp_path)
    relay = _BudgetConsumingRelay(40)
    service.personal_context_relay = relay
    service._recovery_clock_ns = _ManualClock()
    raw_cursors = [
        _insert_hidden_ingress(service, ordinal=index) for index in range(61)
    ]
    cursor: str | int = 0
    if signed:
        device = service._require_registered_device("user-a", "device-b")
        cursor = service._encode_pull_token(
            dataset_id=DATASET_ID,
            device_id="device-b",
            version_set=service._pull_version_set(device),
            watermarks={(DOMAIN, 1): 0},
        )

    pulled = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=[DOMAIN],
        cursor=cursor,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )

    assert len(relay.budgets) == 1
    assert relay.budgets[0].remaining_rows == 0
    if signed:
        device = service._require_registered_device("user-a", "device-b")
        decoded = service._decode_pull_token(
            pulled.next_cursor,
            dataset_id=DATASET_ID,
            device_id="device-b",
            version_set=service._pull_version_set(device),
            streams=[(DOMAIN, 1)],
        )
        assert decoded[(DOMAIN, 1)] == raw_cursors[59]
    else:
        assert pulled.next_cursor == str(raw_cursors[59])
    assert pulled.has_more is True


def _source_row(sequence: int) -> PublicationSourceRow:
    return PublicationSourceRow(
        profile_id="profile-a",
        profile_publication_sequence=sequence,
        publication_batch_id=f"batch-{sequence}",
        batch_ordinal=0,
        batch_size=1,
        purge_generation=0,
        role="semantic",
        object_id=f"object-{sequence}",
        version_id=f"version-{sequence}",
        operation="upsert",
        deterministic_envelope_id=f"authority-{sequence}",
        integrity_tag="hmac-sha256-v1:" + "0" * 64,
        domain=DOMAIN,
        canonical=b"{}",
        sync_server_cursor=None,
        row_state="pending",
    )


class _ManyBatchSource:
    def __init__(
        self,
        rows: int,
        *,
        expire_after_selection: _ManualClock | None = None,
    ) -> None:
        self.remaining = rows
        self.sequence = 0
        self.source_limits: list[int] = []
        self.expire_after_selection = expire_after_selection

    @contextmanager
    def profile_lease(self, _profile_id: str):
        yield SimpleNamespace(owner_token="owner")

    def earliest_nonterminal_batch(
        self,
        _profile_id: str,
        *,
        row_limit: int,
        lease: object,
        budget: _PersonalContextRecoveryBudget,
    ) -> PublicationSourceBatch | None:
        del lease
        self.source_limits.append(row_limit)
        if self.remaining == 0:
            return None
        assert budget.consume()
        self.sequence += 1
        self.remaining -= 1
        row = _source_row(self.sequence)
        if self.expire_after_selection is not None:
            self.expire_after_selection.now_ns = 100
        return PublicationSourceBatch(
            row.profile_id,
            row.profile_publication_sequence,
            row.publication_batch_id,
            (row,),
        )

    def renew_lease(self, _lease: object) -> bool:
        return True

    def row_is_current(self, _row: object, _lease: object) -> bool:
        return True

    def record_staged_row(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def acknowledge_row(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def complete_if_acknowledged(self, *_args: Any, **_kwargs: Any) -> bool:
        return True


class _LeaseFlipSource(_ManyBatchSource):
    """Expire the shared deadline from a successful current-row check."""

    def __init__(
        self,
        *,
        row_state: str,
        expire_on_current_call: int,
        clock: _ManualClock,
    ) -> None:
        super().__init__(1)
        self.row_state = row_state
        self.expire_on_current_call = expire_on_current_call
        self.clock = clock
        self.current_calls = 0
        self.actions: list[str] = []

    def earliest_nonterminal_batch(
        self,
        profile_id: str,
        *,
        row_limit: int,
        lease: object,
        budget: _PersonalContextRecoveryBudget,
    ) -> PublicationSourceBatch | None:
        batch = super().earliest_nonterminal_batch(
            profile_id,
            row_limit=row_limit,
            lease=lease,
            budget=budget,
        )
        if batch is None:
            return None
        row = replace(
            batch.rows[0],
            row_state=self.row_state,
            sync_server_cursor=(1 if self.row_state == "acknowledged" else None),
        )
        return replace(batch, rows=(row,))

    def row_is_current(self, _row: object, _lease: object) -> bool:
        self.current_calls += 1
        if self.current_calls == self.expire_on_current_call:
            self.clock.now_ns = 100_000_000
        return True

    def record_staged_row(self, *_args: Any, **_kwargs: Any) -> None:
        self.actions.append("record")

    def acknowledge_row(self, *_args: Any, **_kwargs: Any) -> None:
        self.actions.append("acknowledge")

    def complete_if_acknowledged(self, *_args: Any, **_kwargs: Any) -> bool:
        self.actions.append("complete")
        return True


def test_relay_never_queries_a_zero_limit_after_exactly_one_hundred_batches() -> None:
    """The old loop queried row_limit=0 after using its final allowance."""

    publications = _ManyBatchSource(101)

    def stage(row: PublicationSourceRow, *_args: Any) -> AuthorityStageReceipt:
        return AuthorityStageReceipt(
            server_cursor=row.profile_publication_sequence,
            deterministic_envelope_id=row.deterministic_envelope_id,
            publication_batch_id=row.publication_batch_id,
            profile_publication_sequence=row.profile_publication_sequence,
            batch_ordinal=0,
            batch_size=1,
            purge_generation=0,
        )

    result = PersonalContextRelay(
        publications=publications,
        stage_authority=stage,
    ).relay_profile(
        user_id="user-a",
        profile_id="profile-a",
        dataset_id="dataset-a",
        after_server_cursor=None,
        row_budget=100,
        wall_time_ms=5_000,
    )

    assert result.continuation == "personal_context_relay_pending"
    assert result.inspected_rows == 100
    assert len(publications.source_limits) == 100
    assert 0 not in publications.source_limits
    assert publications.remaining == 1


def test_relay_does_not_restore_authority_after_source_selection_expires() -> None:
    """A selected row still needs a deadline fence before authority restoration."""

    clock = _ManualClock()
    publications = _ManyBatchSource(1, expire_after_selection=clock)
    staged: list[PublicationSourceRow] = []
    budget = _PersonalContextRecoveryBudget(
        deadline_ns=100,
        remaining_rows=100,
        clock_ns=clock,
    )

    def stage(row: PublicationSourceRow, *_args: Any) -> AuthorityStageReceipt:
        staged.append(row)
        return AuthorityStageReceipt(
            server_cursor=1,
            deterministic_envelope_id=row.deterministic_envelope_id,
            publication_batch_id=row.publication_batch_id,
            profile_publication_sequence=row.profile_publication_sequence,
            batch_ordinal=0,
            batch_size=1,
            purge_generation=0,
        )

    result = PersonalContextRelay(
        publications=publications,
        stage_authority=stage,
    ).relay_profile(
        user_id="user-a",
        profile_id="profile-a",
        dataset_id="dataset-a",
        after_server_cursor=None,
        budget=budget,
    )

    assert result.continuation == "personal_context_relay_pending"
    assert result.inspected_rows == 1
    assert staged == []


@pytest.mark.parametrize(
    ("row_state", "expire_on_current_call", "expected_actions"),
    [
        pytest.param("pending", 1, [], id="before-stage"),
        pytest.param("pending", 2, ["stage"], id="before-record"),
        pytest.param(
            "pending",
            3,
            ["stage", "record"],
            id="before-acknowledge",
        ),
        pytest.param(
            "pending",
            4,
            ["stage", "record", "acknowledge"],
            id="before-finalize",
        ),
        pytest.param("acknowledged", 1, [], id="acknowledged-before-finalize"),
    ],
)
def test_relay_rechecks_deadline_after_each_successful_current_row_check(
    row_state: str,
    expire_on_current_call: int,
    expected_actions: list[str],
) -> None:
    """Lease/current validation may itself consume the remaining wall time."""

    clock = _ManualClock()
    publications = _LeaseFlipSource(
        row_state=row_state,
        expire_on_current_call=expire_on_current_call,
        clock=clock,
    )
    budget = _PersonalContextRecoveryBudget(
        deadline_ns=100,
        remaining_rows=100,
        clock_ns=clock,
    )

    def stage(row: PublicationSourceRow, *_args: Any) -> AuthorityStageReceipt:
        publications.actions.append("stage")
        return AuthorityStageReceipt(
            server_cursor=1,
            deterministic_envelope_id=row.deterministic_envelope_id,
            publication_batch_id=row.publication_batch_id,
            profile_publication_sequence=row.profile_publication_sequence,
            batch_ordinal=row.batch_ordinal,
            batch_size=row.batch_size,
            purge_generation=row.purge_generation,
        )

    def finalize(*_args: Any) -> None:
        publications.actions.append("finalize")

    result = PersonalContextRelay(
        publications=publications,
        stage_authority=stage,
        finalize_authority=finalize,
    ).relay_profile(
        user_id="user-a",
        profile_id="profile-a",
        dataset_id="dataset-a",
        after_server_cursor=None,
        budget=budget,
    )

    assert result.continuation == "personal_context_relay_pending"
    assert publications.actions == expected_actions


def test_row_current_deadline_expiry_cannot_advance_pull_watermark(tmp_path) -> None:
    """A relay deadline race cannot hand later raw rows a fresh clock window."""

    service, _target, _sqlite_path = _service(tmp_path)
    clock = _ManualClock()
    publications = _LeaseFlipSource(
        row_state="pending",
        expire_on_current_call=1,
        clock=clock,
    )
    service._recovery_clock_ns = clock
    service.personal_context_relay = PersonalContextRelay(
        publications=publications,
        stage_authority=lambda *_args: publications.actions.append("stage"),
    )
    _insert_hidden_ingress(service, ordinal=1)

    pulled = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=[DOMAIN],
        cursor=0,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )

    assert pulled.next_cursor == "0"
    assert pulled.has_more is True
    assert publications.actions == []


def test_relay_rechecks_deadline_before_completing_the_batch() -> None:
    """Finalization may consume the remaining time before the batch-state write."""

    clock = _ManualClock()
    publications = _LeaseFlipSource(
        row_state="pending",
        expire_on_current_call=99,
        clock=clock,
    )
    budget = _PersonalContextRecoveryBudget(
        deadline_ns=100,
        remaining_rows=100,
        clock_ns=clock,
    )

    def stage(row: PublicationSourceRow, *_args: Any) -> AuthorityStageReceipt:
        publications.actions.append("stage")
        return AuthorityStageReceipt(
            server_cursor=1,
            deterministic_envelope_id=row.deterministic_envelope_id,
            publication_batch_id=row.publication_batch_id,
            profile_publication_sequence=row.profile_publication_sequence,
            batch_ordinal=row.batch_ordinal,
            batch_size=row.batch_size,
            purge_generation=row.purge_generation,
        )

    def finalize(*_args: Any) -> None:
        publications.actions.append("finalize")
        clock.now_ns = 100

    result = PersonalContextRelay(
        publications=publications,
        stage_authority=stage,
        finalize_authority=finalize,
    ).relay_profile(
        user_id="user-a",
        profile_id="profile-a",
        dataset_id="dataset-a",
        after_server_cursor=None,
        budget=budget,
    )

    assert result.continuation == "personal_context_relay_pending"
    assert publications.actions == [
        "stage",
        "record",
        "acknowledge",
        "finalize",
    ]


def test_real_source_decryption_stops_when_deadline_expires_between_rows(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checking only before a batch would decrypt rows after the deadline."""

    runtime = AuthorityHarness(tmp_path, monkeypatch)
    clock = _ManualClock()
    budget = _PersonalContextRecoveryBudget(
        deadline_ns=100,
        remaining_rows=100,
        clock_ns=clock,
    )
    decrypt_calls = 0
    original = PersonalContextPublicationJournal.decrypt_row

    def expire_after_first_decrypt(journal: Any, row: Any):
        nonlocal decrypt_calls
        decrypt_calls += 1
        result = original(journal, row)
        clock.now_ns = 100
        return result

    monkeypatch.setattr(
        PersonalContextPublicationJournal,
        "decrypt_row",
        expire_after_first_decrypt,
    )
    with runtime.publications.profile_lease(runtime.manifest.profile_id) as lease:
        assert lease is not None
        batch = runtime.publications.earliest_nonterminal_batch(
            runtime.manifest.profile_id,
            row_limit=100,
            lease=lease,
            budget=budget,
        )

    assert batch is not None
    assert len(batch.rows) == 1
    assert decrypt_calls == 1
    assert budget.remaining_rows == 99


def test_raw_scan_rechecks_deadline_after_each_classification(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A receipt lookup that crosses the deadline cannot advance its row."""

    service, _target, _sqlite_path = _service(tmp_path)
    cursors = [_insert_hidden_ingress(service, ordinal=index) for index in range(3)]
    clock = _ManualClock()
    budget = _PersonalContextRecoveryBudget(
        deadline_ns=100,
        remaining_rows=100,
        clock_ns=clock,
    )
    original_classify = service.store.classify_personal_context_recovery_row
    classify_calls = 0

    def expire_during_second_classification(*args: Any, **kwargs: Any):
        nonlocal classify_calls
        classify_calls += 1
        result = original_classify(*args, **kwargs)
        if classify_calls == 2:
            clock.now_ns = 100
        return result

    monkeypatch.setattr(
        service.store,
        "classify_personal_context_recovery_row",
        expire_during_second_classification,
    )

    scan = service.store.scan_personal_context_authority(
        DATASET_ID,
        after_server_cursor=0,
        limit=10,
        budget=budget,
        domains=[DOMAIN],
        adapter_versions=[1],
        profile_id="profile-a",
        integrity_key_id="personal-context-integrity-v1",
        purge_generation=0,
    )

    assert scan.raw_rows_scanned == 2
    assert scan.raw_scan_watermark == cursors[0]
    assert scan.source_exhausted is False
    assert budget.remaining_rows == 98


@pytest.mark.parametrize("signed", [False, True])
def test_deadline_before_authority_decrypt_advances_only_hidden_prefix(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    signed: bool,
) -> None:
    """A scan watermark cannot pass authority that was never decrypted."""

    service, _target, _sqlite_path = _service(tmp_path)
    relay = _BudgetConsumingRelay(0)
    service.personal_context_relay = relay
    clock = _ManualClock()
    service._recovery_clock_ns = clock
    hidden_cursor = _insert_hidden_ingress(service, ordinal=1)
    _insert_authority(service, record_id="deadline-authority", sequence=1)
    cursor: str | int = 0
    scan_name = "scan_personal_context_authority"
    scan_owner: Any = service.store
    if signed:
        device = service._require_registered_device("user-a", "device-b")
        cursor = service._encode_pull_token(
            dataset_id=DATASET_ID,
            device_id="device-b",
            version_set=service._pull_version_set(device),
            watermarks={(DOMAIN, 1): 0},
        )
        scan_name = "_scan_versioned_pull_page"
        scan_owner = service
    original_scan = getattr(scan_owner, scan_name)

    def expire_after_scan(*args: Any, **kwargs: Any):
        result = original_scan(*args, **kwargs)
        clock.now_ns = 100_000_000
        return result

    monkeypatch.setattr(scan_owner, scan_name, expire_after_scan)
    original_restore = service._restore_personal_context_from_storage

    def fail_if_decrypted(*args: Any, **kwargs: Any):
        raise AssertionError("authority decrypted after the recovery deadline")

    monkeypatch.setattr(
        service,
        "_restore_personal_context_from_storage",
        fail_if_decrypted,
    )
    try:
        pulled = service.pull(
            user_id="user-a",
            dataset_id=DATASET_ID,
            device_id="device-b",
            domains=[DOMAIN],
            cursor=cursor,
            page_size=10,
            include_own_changes=True,
            personal_context_exchange=EXCHANGE,
        )
    finally:
        monkeypatch.setattr(
            service,
            "_restore_personal_context_from_storage",
            original_restore,
        )

    assert pulled.envelopes == []
    assert pulled.has_more is True
    if signed:
        device = service._require_registered_device("user-a", "device-b")
        decoded = service._decode_pull_token(
            pulled.next_cursor,
            dataset_id=DATASET_ID,
            device_id="device-b",
            version_set=service._pull_version_set(device),
            streams=[(DOMAIN, 1)],
        )
        assert decoded[(DOMAIN, 1)] == hidden_cursor
    else:
        assert pulled.next_cursor == str(hidden_cursor)


def test_page_lookahead_cannot_borrow_a_hundred_and_first_row(tmp_path) -> None:
    """Page-plus-one is recovery work and must fit the remaining allowance."""

    service, _target, _sqlite_path = _service(tmp_path)
    relay = _BudgetConsumingRelay(99)
    service.personal_context_relay = relay
    service._recovery_clock_ns = _ManualClock()
    first_cursor = _insert_authority(service, record_id="first", sequence=1)
    _insert_authority(service, record_id="second", sequence=2)

    pulled = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=[DOMAIN],
        cursor=0,
        page_size=1,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )

    assert [item.server_cursor for item in pulled.envelopes] == [first_cursor]
    assert relay.budgets[0].remaining_rows == 0
    assert pulled.next_cursor == str(first_cursor)
    assert pulled.has_more is True


def test_current_unclassified_personal_context_row_is_a_watermark_barrier(
    tmp_path,
) -> None:
    """An ineligible row is skippable only after a permanent hidden role is proven."""

    service, _target, _sqlite_path = _service(tmp_path)
    cursor = _insert_hidden_ingress(service, ordinal=1, attested=False)
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            "UPDATE sync_envelopes SET routing_metadata_json = ? WHERE server_sequence = ?",
            (
                '{"integrity_key_id":"personal-context-integrity-v1",'
                '"profile_id":"profile-a","purge_generation":0}',
                cursor,
            ),
            connection=connection,
        )
    budget = _PersonalContextRecoveryBudget(
        deadline_ns=100,
        remaining_rows=100,
        clock_ns=_ManualClock(),
    )

    scan = service.store.scan_personal_context_authority(
        DATASET_ID,
        after_server_cursor=0,
        limit=10,
        budget=budget,
        domains=[DOMAIN],
        adapter_versions=[1],
        profile_id="profile-a",
        integrity_key_id="personal-context-integrity-v1",
        purge_generation=0,
    )

    assert scan.raw_rows_scanned == 1
    assert scan.raw_scan_watermark == 0
    assert scan.visible_envelopes == []
    assert scan.source_exhausted is False


def test_conflict_barrier_advances_only_the_inspected_hidden_prefix(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conflict row may consume budget but cannot become a safe watermark."""

    service, _target, _sqlite_path = _service(tmp_path)
    hidden_cursor = _insert_hidden_ingress(service, ordinal=1)
    conflict_cursor = _insert_authority(
        service,
        record_id="conflict-barrier",
        sequence=1,
    )
    monkeypatch.setattr(
        service.store,
        "get_unresolved_materialization_conflict",
        lambda _dataset_id: SimpleNamespace(
            server_sequence=conflict_cursor,
            conflict_type="object_revision",
        ),
    )
    budget = _PersonalContextRecoveryBudget(
        deadline_ns=100,
        remaining_rows=100,
        clock_ns=_ManualClock(),
    )

    scan = service.store.scan_personal_context_authority(
        DATASET_ID,
        after_server_cursor=0,
        limit=10,
        budget=budget,
        domains=[DOMAIN],
        adapter_versions=[1],
        profile_id="profile-a",
        integrity_key_id="personal-context-integrity-v1",
        purge_generation=0,
    )

    assert scan.raw_rows_scanned == 2
    assert scan.raw_scan_watermark == hidden_cursor
    assert scan.visible_envelopes == []
    assert scan.source_exhausted is False
    assert budget.remaining_rows == 98


def _tampered_home_authority_routing(case: str) -> object:
    routing: dict[str, object] = {
        "integrity_key_id": "personal-context-integrity-v1",
        "profile_id": "profile-a",
        "purge_generation": 0,
        "personal_context_authority": {
            "role": "home_authority",
            "publication_batch_id": "publication-batch-0001",
            "profile_publication_sequence": 1,
            "batch_ordinal": 0,
            "batch_size": 1,
        },
    }
    if case == "missing":
        routing.pop("profile_id")
    elif case == "malformed":
        routing["profile_id"] = []
    elif case == "generation-type-drift":
        routing["purge_generation"] = "0"
    elif case == "role-tamper":
        routing["personal_context_authority"] = {"role": "client_ingress"}
    else:  # pragma: no cover - the parametrization is closed.
        raise AssertionError(f"unknown tamper case: {case}")
    return routing


@pytest.mark.parametrize("signed", [False, True], ids=["legacy", "signed"])
@pytest.mark.parametrize(
    "tamper_case",
    ["missing", "malformed", "generation-type-drift", "role-tamper"],
)
def test_tampered_home_authority_never_enters_the_safe_watermark_prefix(
    tmp_path,
    signed: bool,
    tamper_case: str,
) -> None:
    """Mutable routing cannot relabel applied home authority as safe hidden ingress."""

    service, _target, _sqlite_path = _service(tmp_path)
    service.personal_context_relay = _BudgetConsumingRelay(0)
    service._recovery_clock_ns = _ManualClock()
    authority_cursor = _insert_authority(
        service,
        record_id="tampered-authority",
        sequence=1,
    )
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            "UPDATE sync_envelopes SET routing_metadata_json = ? "
            "WHERE server_sequence = ?",
            (
                json.dumps(
                    _tampered_home_authority_routing(tamper_case),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                authority_cursor,
            ),
            connection=connection,
        )

    cursor: str | int = 0
    if signed:
        device = service._require_registered_device("user-a", "device-b")
        cursor = service._encode_pull_token(
            dataset_id=DATASET_ID,
            device_id="device-b",
            version_set=service._pull_version_set(device),
            watermarks={(DOMAIN, 1): 0},
        )

    pulled = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=[DOMAIN],
        cursor=cursor,
        include_own_changes=True,
        personal_context_exchange=EXCHANGE,
    )

    assert pulled.envelopes == []
    assert pulled.has_more is True
    if signed:
        device = service._require_registered_device("user-a", "device-b")
        decoded = service._decode_pull_token(
            pulled.next_cursor,
            dataset_id=DATASET_ID,
            device_id="device-b",
            version_set=service._pull_version_set(device),
            streams=[(DOMAIN, 1)],
        )
        assert decoded[(DOMAIN, 1)] == 0
    else:
        assert pulled.next_cursor == "0"


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("dataset_id", "dataset-tampered"),
        ("device_id", "device-tampered"),
        ("client_envelope_id", "ingress-tampered"),
        ("wire_entity_version", "version-tampered"),
    ],
)
def test_hidden_ingress_requires_an_exact_receipt_identity(
    tmp_path,
    column: str,
    value: str,
) -> None:
    """A receipt for a different ingress identity cannot authorize a safe skip."""

    service, _target, _sqlite_path = _service(tmp_path)
    cursor = _insert_hidden_ingress(service, ordinal=1)
    statements = {
        "dataset_id": (
            "UPDATE sync_personal_context_ingress_receipts SET dataset_id = ? "
            "WHERE server_sequence = ?"
        ),
        "device_id": (
            "UPDATE sync_personal_context_ingress_receipts SET device_id = ? "
            "WHERE server_sequence = ?"
        ),
        "client_envelope_id": (
            "UPDATE sync_personal_context_ingress_receipts "
            "SET client_envelope_id = ? WHERE server_sequence = ?"
        ),
        "wire_entity_version": (
            "UPDATE sync_personal_context_ingress_receipts "
            "SET wire_entity_version = ? WHERE server_sequence = ?"
        ),
    }
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            statements[column],
            (value, cursor),
            connection=connection,
        )
    budget = _PersonalContextRecoveryBudget(
        deadline_ns=100,
        remaining_rows=100,
        clock_ns=_ManualClock(),
    )

    scan = service.store.scan_personal_context_authority(
        DATASET_ID,
        after_server_cursor=0,
        limit=10,
        budget=budget,
        domains=[DOMAIN],
        adapter_versions=[1],
        profile_id="profile-a",
        integrity_key_id="personal-context-integrity-v1",
        purge_generation=0,
    )

    assert scan.raw_scan_watermark == 0
    assert scan.visible_envelopes == []
    assert scan.source_exhausted is False


def test_attested_ingress_remains_safe_when_mutable_routing_is_stale(tmp_path) -> None:
    """The exact canonical receipt, not mutable routing, proves client ingress."""

    service, _target, _sqlite_path = _service(tmp_path)
    cursor = _insert_hidden_ingress(service, ordinal=1)
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            "UPDATE sync_envelopes SET routing_metadata_json = '{}' "
            "WHERE server_sequence = ?",
            (cursor,),
            connection=connection,
        )
    budget = _PersonalContextRecoveryBudget(
        deadline_ns=100,
        remaining_rows=100,
        clock_ns=_ManualClock(),
    )

    scan = service.store.scan_personal_context_authority(
        DATASET_ID,
        after_server_cursor=0,
        limit=10,
        budget=budget,
        domains=[DOMAIN],
        adapter_versions=[1],
        profile_id="profile-a",
        integrity_key_id="personal-context-integrity-v1",
        purge_generation=1,
    )

    assert scan.raw_scan_watermark == cursor
    assert scan.visible_envelopes == []
    assert scan.source_exhausted is True


def test_shredded_routing_marker_alone_cannot_hide_authority(tmp_path) -> None:
    """Mutable retention metadata is insufficient without cleanup's empty shape."""

    service, _target, _sqlite_path = _service(tmp_path)
    cursor = _insert_authority(service, record_id="marker-only", sequence=1)
    stored = service.store.get_envelope_by_server_cursor(cursor)
    assert stored is not None
    routing = dict(stored.routing_metadata)
    routing["retention_state"] = "shredded"
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            "UPDATE sync_envelopes SET routing_metadata_json = ? "
            "WHERE server_sequence = ?",
            (json.dumps(routing, sort_keys=True), cursor),
            connection=connection,
        )
    budget = _PersonalContextRecoveryBudget(
        deadline_ns=100,
        remaining_rows=100,
        clock_ns=_ManualClock(),
    )

    scan = service.store.scan_personal_context_authority(
        DATASET_ID,
        after_server_cursor=0,
        limit=10,
        budget=budget,
        domains=[DOMAIN],
        adapter_versions=[1],
        profile_id="profile-a",
        integrity_key_id="personal-context-integrity-v1",
        purge_generation=0,
    )

    assert [item.server_cursor for item in scan.visible_envelopes] == [cursor]
    assert scan.raw_scan_watermark == cursor


def test_structurally_shredded_authority_is_safe_to_skip(tmp_path) -> None:
    """Cleanup's content-free row shape may advance an old-generation watermark."""

    service, _target, _sqlite_path = _service(tmp_path)
    cursor = _insert_authority(service, record_id="shredded", sequence=1)
    routing = json.dumps(
        {
            "profile_id": "profile-a",
            "purge_generation": 0,
            "retention_state": "shredded",
        },
        sort_keys=True,
    )
    with service.store.db.backend.transaction() as connection:
        service.store.db.execute(
            """UPDATE sync_envelopes
               SET stable_key = NULL, mutation_group_id = NULL,
                   mutation_step = NULL, mutation_step_count = NULL,
                   mutation_plan_hash = NULL, base_object_hash = NULL,
                   base_version = NULL, entity_version = NULL,
                   dependency_json = '[]', routing_metadata_json = ?,
                   payload_ciphertext = NULL, payload_json = '{}',
                   payload_clear_json = '{}', payload_hash = NULL,
                   payload_size_bytes = 0, encryption_metadata_json = '{}'
               WHERE server_sequence = ?""",
            (routing, cursor),
            connection=connection,
        )
    budget = _PersonalContextRecoveryBudget(
        deadline_ns=100,
        remaining_rows=100,
        clock_ns=_ManualClock(),
    )

    scan = service.store.scan_personal_context_authority(
        DATASET_ID,
        after_server_cursor=0,
        limit=10,
        budget=budget,
        domains=[DOMAIN],
        adapter_versions=[1],
        profile_id="profile-a",
        integrity_key_id="personal-context-integrity-v1",
        purge_generation=1,
    )

    assert scan.raw_scan_watermark == cursor
    assert scan.visible_envelopes == []
    assert scan.source_exhausted is True
