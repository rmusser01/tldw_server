from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Slides import (
    standalone_html_reconciler as reconciler_module,
)
from tldw_Server_API.app.core.Slides.slides_db import SlidesDatabase
from tldw_Server_API.app.core.Slides.standalone_html_reconciler import (
    FencedStandaloneHtmlReconciler,
    ReconciliationCursor,
    UnsafeSlidesDatabaseError,
    decode_reconciliation_cursor,
    discover_canonical_slides_databases,
    encode_reconciliation_cursor,
    reconcile_owner_generation_receipts,
    reconcile_owner_local_expiry,
    reconciliation_admission_ready,
)

UTC = timezone.utc


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _insert_generation_receipt(
    db: SlidesDatabase,
    *,
    receipt_id: str,
    owner_user_id: str = "42",
    status: str = "claimed",
    job_id: int | None = None,
    job_uuid: str | None = None,
    presentation_id: str | None = None,
    input_expires_at: str = "2026-07-19T12:00:00+00:00",
    include_input: bool = True,
) -> str:
    jobs_key = f"slides:v1:{_sha256(f'jobs:{receipt_id}')}"
    with db.transaction(immediate=True) as connection:
        connection.execute(
            """
            INSERT INTO slides_generation_receipts (
                id, owner_user_id, digest_key_id,
                idempotency_key_hmac_sha256, jobs_idempotency_key,
                client_request_hmac_sha256, execution_hmac_sha256,
                job_id, job_uuid, presentation_id, receipt_status,
                error_code, error_message, created_at, updated_at, expires_at
            ) VALUES (?, ?, 'key-v1', ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, NULL)
            """,
            (
                receipt_id,
                owner_user_id,
                _sha256(f"idempotency:{receipt_id}"),
                jobs_key,
                _sha256(f"request:{receipt_id}"),
                _sha256(f"execution:{receipt_id}"),
                job_id,
                job_uuid,
                presentation_id,
                status,
                "2026-07-18T12:00:00+00:00",
                "2026-07-18T12:00:00+00:00",
            ),
        )
        if include_input:
            connection.execute(
                """
                INSERT INTO slides_generation_inputs (
                    receipt_id, source_kind, source_text, source_hmac_sha256,
                    source_bytes, provenance_json, html_options_json, provider,
                    model, adapter_id, endpoint_identity, system_prompt,
                    prompt_sha256, prompt_contract_version, input_expires_at,
                    created_at
                ) VALUES (?, 'prompt', ?, ?, ?, '{}', '{}', 'openai',
                          'gpt-test', 'openai_official_chat_v1',
                          'https://api.openai.com:443/v1/chat/completions',
                          ?, ?, 'slides.standalone_html.v1', ?, ?)
                """,
                (
                    receipt_id,
                    f"SOURCE-SECRET:{receipt_id}",
                    _sha256(f"SOURCE-SECRET:{receipt_id}"),
                    len(f"SOURCE-SECRET:{receipt_id}".encode()),
                    f"PROMPT-SECRET:{receipt_id}",
                    _sha256(f"PROMPT-SECRET:{receipt_id}"),
                    input_expires_at,
                    "2026-07-18T12:00:00+00:00",
                ),
            )
    return jobs_key


def _job(
    *,
    receipt_id: str,
    jobs_key: str,
    status: str,
    job_uuid: str,
    job_id: int | None = 7,
    archived: bool = False,
) -> dict[str, object]:
    return {
        "id": job_id,
        "uuid": job_uuid,
        "owner_user_id": "42",
        "domain": "slides",
        "queue": "default",
        "job_type": "presentation.generate",
        "idempotency_key": jobs_key,
        "payload": {"receipt_id": receipt_id},
        "status": status,
        "archived": archived,
        "error_code": "generation_provider_failed" if status == "failed" else None,
    }


class _JobsStore:
    def __init__(
        self,
        job: dict[str, object] | None = None,
        *,
        error: Exception | None = None,
        terminalization_outcomes: tuple[str, ...] = (),
    ) -> None:
        self.job = job
        self.error = error
        self.terminalization_outcomes = list(terminalization_outcomes)
        self.lookups: list[dict[str, object]] = []
        self.terminalizations: list[dict[str, object]] = []

    def lookup_slides_generation_job(self, **kwargs):
        self.lookups.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return self.job

    def terminalize_slides_generation_job_from_reconciler(self, **kwargs):
        self.terminalizations.append(dict(kwargs))
        return self.terminalization_outcomes.pop(0) if self.terminalization_outcomes else "APPLIED"


def _create_slides_db(base_dir: Path, owner_user_id: str) -> Path:
    user_dir = base_dir / owner_user_id
    user_dir.mkdir(parents=True)
    db_path = user_dir / "Slides.db"
    db = SlidesDatabase(db_path=db_path, client_id=owner_user_id)
    db.close_connection()
    return db_path


def test_discovery_is_numeric_one_level_deterministic_and_cursor_bounded(tmp_path: Path) -> None:
    base_dir = tmp_path / "user_databases"
    _create_slides_db(base_dir, "10")
    _create_slides_db(base_dir, "2")
    _create_slides_db(base_dir, "20")
    _create_slides_db(base_dir / "nested", "3")
    _create_slides_db(base_dir, "alpha")
    _create_slides_db(base_dir, "01")

    first = discover_canonical_slides_databases(
        base_dir,
        after_owner_user_id=None,
        limit=2,
    )
    second = discover_canonical_slides_databases(
        base_dir,
        after_owner_user_id=first[-1].owner_user_id,
        limit=2,
    )

    assert tuple(item.owner_user_id for item in first) == ("2", "10")
    assert tuple(item.owner_user_id for item in second) == ("20",)
    assert all(item.path == base_dir / item.owner_user_id / "Slides.db" for item in (*first, *second))


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_discovery_fails_closed_for_canonical_symlinked_database(tmp_path: Path) -> None:
    base_dir = tmp_path / "user_databases"
    target = _create_slides_db(base_dir, "2")
    unsafe_dir = base_dir / "3"
    unsafe_dir.mkdir()
    (unsafe_dir / "Slides.db").symlink_to(target)

    with pytest.raises(UnsafeSlidesDatabaseError) as exc_info:
        discover_canonical_slides_databases(
            base_dir,
            after_owner_user_id=None,
            limit=10,
        )

    assert str(exc_info.value) == "standalone_html_slides_database_unsafe"


def test_discovery_fails_closed_for_incomplete_canonical_schema(tmp_path: Path) -> None:
    base_dir = tmp_path / "user_databases"
    user_dir = base_dir / "7"
    user_dir.mkdir(parents=True)
    (user_dir / "Slides.db").touch()

    with pytest.raises(UnsafeSlidesDatabaseError) as exc_info:
        discover_canonical_slides_databases(
            base_dir,
            after_owner_user_id=None,
            limit=10,
        )

    assert str(exc_info.value) == "standalone_html_slides_database_unsafe"


def test_discovery_does_not_repair_a_v2_database_missing_a_base_schema_object(
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "user_databases"
    db_path = _create_slides_db(base_dir, "7")
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE sync_log")

    with pytest.raises(UnsafeSlidesDatabaseError) as exc_info:
        discover_canonical_slides_databases(
            base_dir,
            after_owner_user_id=None,
            limit=10,
        )

    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as connection:
        sync_log = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sync_log'"
        ).fetchone()
    assert str(exc_info.value) == "standalone_html_slides_database_unsafe"
    assert sync_log is None


def test_discovery_does_not_create_a_missing_registry_root(tmp_path: Path) -> None:
    base_dir = tmp_path / "missing"

    assert (
        discover_canonical_slides_databases(
            base_dir,
            after_owner_user_id=None,
            limit=10,
        )
        == ()
    )
    assert not base_dir.exists()


def test_discovery_page_counts_databases_not_empty_user_directories(tmp_path: Path) -> None:
    base_dir = tmp_path / "user_databases"
    (base_dir / "1").mkdir(parents=True)
    (base_dir / "2").mkdir()
    _create_slides_db(base_dir, "10")

    discovered = discover_canonical_slides_databases(
        base_dir,
        after_owner_user_id=None,
        limit=1,
    )

    assert tuple(item.owner_user_id for item in discovered) == ("10",)


def test_discovery_fails_closed_when_selected_database_disappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_dir = tmp_path / "user_databases"
    db_path = _create_slides_db(base_dir, "2")
    original_stat = reconciler_module.os.stat
    candidate_stat_calls = 0

    def disappearing_stat(path, *args, **kwargs):
        nonlocal candidate_stat_calls
        if Path(path) == db_path and kwargs.get("follow_symlinks") is False:
            candidate_stat_calls += 1
            if candidate_stat_calls == 2:
                db_path.unlink()
                raise FileNotFoundError(db_path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(reconciler_module.os, "stat", disappearing_stat)

    with pytest.raises(UnsafeSlidesDatabaseError):
        discover_canonical_slides_databases(
            base_dir,
            after_owner_user_id=None,
            limit=1,
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_discovery_fails_closed_for_symlinked_registry_root(tmp_path: Path) -> None:
    real_base_dir = tmp_path / "real_user_databases"
    _create_slides_db(real_base_dir, "2")
    symlinked_base_dir = tmp_path / "user_databases"
    symlinked_base_dir.symlink_to(real_base_dir, target_is_directory=True)

    with pytest.raises(UnsafeSlidesDatabaseError):
        discover_canonical_slides_databases(
            symlinked_base_dir,
            after_owner_user_id=None,
            limit=1,
        )


@pytest.mark.parametrize(
    "invalid_now",
    [
        pytest.param(datetime(2026, 7, 18, 12, 0), id="naive"),
        pytest.param(
            datetime(
                2026,
                7,
                18,
                13,
                0,
                tzinfo=timezone(timedelta(hours=1)),
            ),
            id="non-utc-offset",
        ),
    ],
)
def test_owner_reconciliation_rejects_non_utc_now(
    tmp_path: Path,
    invalid_now: datetime,
) -> None:
    db_path = _create_slides_db(tmp_path / "user_databases", "2")
    db = SlidesDatabase(db_path=db_path, client_id="2")

    try:
        with pytest.raises(ValueError, match="aware UTC"):
            reconcile_owner_generation_receipts(
                db,
                _JobsStore(),
                owner_user_id="2",
                now=invalid_now,
                after_receipt_id=None,
                limit=1,
            )
    finally:
        db.close_connection()


def test_reconciliation_cursor_round_trips_only_source_free_progress() -> None:
    cursor = ReconciliationCursor(
        phase="dormant",
        after_owner_user_id="42",
        owner_user_id="43",
        after_receipt_id="018f7f65-a60f-7c21-b690-0bca9205f44f",
    )

    encoded = encode_reconciliation_cursor(cursor)

    assert len(encoded) <= 1024
    assert decode_reconciliation_cursor(encoded) == cursor
    assert "source" not in encoded
    assert "prompt" not in encoded
    assert "html" not in encoded


def test_reconciliation_cursor_preserves_active_and_dormant_phases() -> None:
    active = ReconciliationCursor(phase="active", after_owner_user_id="10")
    dormant = ReconciliationCursor(phase="dormant", after_owner_user_id="10")

    assert decode_reconciliation_cursor(encode_reconciliation_cursor(active)) == active
    assert decode_reconciliation_cursor(encode_reconciliation_cursor(dormant)) == dormant
    assert encode_reconciliation_cursor(active) != encode_reconciliation_cursor(dormant)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "{}",
        '{"v":2,"after_owner_user_id":null,"owner_user_id":null,"after_receipt_id":null}',
        '{"v":1,"after_owner_user_id":"../1","owner_user_id":null,"after_receipt_id":null}',
        '{"v":1,"after_owner_user_id":null,"owner_user_id":"01","after_receipt_id":null}',
        '{"v":1,"after_owner_user_id":null,"owner_user_id":"1","after_receipt_id":"not-a-uuid"}',
        '{"v":1,"phase":"invalid","after_owner_user_id":null,"owner_user_id":null,"after_receipt_id":null}',
    ],
)
def test_reconciliation_cursor_rejects_malformed_or_noncanonical_state(raw: str) -> None:
    with pytest.raises(ValueError, match="reconciliation cursor is invalid"):
        decode_reconciliation_cursor(raw)


def _ready_state(now: datetime) -> dict[str, object]:
    return {
        "holder_uuid": "leader-1",
        "lease_expires_at": now + timedelta(seconds=60),
        "config_revision": "epoch-1",
        "startup_complete_epoch": "epoch-1",
        "last_complete_epoch": now.timestamp() - 60,
        "lag": 60,
    }


def test_admission_requires_live_leader_current_epoch_and_recent_complete_pass() -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)

    assert reconciliation_admission_ready(
        _ready_state(now),
        config_epoch="epoch-1",
        now=now,
    )

    stale = _ready_state(now)
    stale["last_complete_epoch"] = now.timestamp() - 901
    stale["lag"] = 0
    assert not reconciliation_admission_ready(stale, config_epoch="epoch-1", now=now)

    expired_leader = _ready_state(now)
    expired_leader["lease_expires_at"] = now
    assert not reconciliation_admission_ready(expired_leader, config_epoch="epoch-1", now=now)

    wrong_epoch = _ready_state(now)
    wrong_epoch["startup_complete_epoch"] = "epoch-0"
    assert not reconciliation_admission_ready(wrong_epoch, config_epoch="epoch-1", now=now)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("holder_uuid", None),
        ("lease_expires_at", None),
        ("config_revision", "epoch-0"),
        ("startup_complete_epoch", None),
        ("last_complete_epoch", None),
        ("lag", 901),
    ],
)
def test_admission_fails_closed_for_incomplete_shared_state(field: str, value: object) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    state = _ready_state(now)
    state[field] = value

    assert not reconciliation_admission_ready(
        state,
        config_epoch="epoch-1",
        now=now,
    )


def test_owner_reconciliation_enforces_absolute_expiry_when_jobs_is_unavailable(tmp_path: Path) -> None:
    db = SlidesDatabase(tmp_path / "Slides.db", client_id="42")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f44f"
    job_uuid = "018f7f65-a60f-7c21-b690-0bca9205f450"
    _insert_generation_receipt(
        db,
        receipt_id=receipt_id,
        status="queued",
        job_id=7,
        job_uuid=job_uuid,
    )
    jobs = _JobsStore(error=RuntimeError("private jobs database path"))

    result = reconcile_owner_generation_receipts(
        db,
        jobs,
        owner_user_id="42",
        now=datetime(2026, 7, 20, 12, 0, tzinfo=UTC),
        after_receipt_id=None,
        limit=100,
    )

    receipt = db.get_generation_receipt(receipt_id, owner_user_id="42")
    assert result.jobs_available is False
    assert (receipt.receipt_status, receipt.error_code) == ("failed", "generation_expired")
    assert receipt.updated_at == "2026-07-19T12:00:00+00:00"
    assert receipt.expires_at == "2026-08-18T12:00:00+00:00"
    with pytest.raises(KeyError, match="generation_input_not_found"):
        db.get_generation_input(receipt_id, owner_user_id="42")
    assert jobs.terminalizations == []
    db.close_connection()


@pytest.mark.parametrize("local_only", [False, True])
def test_reconciliation_derives_deadline_instead_of_trusting_future_input_metadata(
    tmp_path: Path,
    local_only: bool,
) -> None:
    db = SlidesDatabase(tmp_path / "Slides.db", client_id="42")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f49f"
    _insert_generation_receipt(
        db,
        receipt_id=receipt_id,
        status="queued",
        job_id=7,
        job_uuid="018f7f65-a60f-7c21-b690-0bca9205f4a0",
        input_expires_at="2030-07-19T12:00:00+00:00",
    )
    now = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)

    if local_only:
        reconcile_owner_local_expiry(
            db,
            owner_user_id="42",
            now=now,
            after_receipt_id=None,
            limit=100,
        )
    else:
        reconcile_owner_generation_receipts(
            db,
            _JobsStore(error=RuntimeError("private Jobs outage")),
            owner_user_id="42",
            now=now,
            after_receipt_id=None,
            limit=100,
        )

    receipt = db.get_generation_receipt(receipt_id, owner_user_id="42")
    assert (receipt.receipt_status, receipt.error_code) == (
        "failed",
        "generation_expired",
    )
    assert receipt.updated_at == "2026-07-19T12:00:00+00:00"
    with pytest.raises(KeyError, match="generation_input_not_found"):
        db.get_generation_input(receipt_id, owner_user_id="42")
    db.close_connection()


def test_expired_receipt_retries_jobs_terminalization_after_outage_recovers(tmp_path: Path) -> None:
    db = SlidesDatabase(tmp_path / "Slides.db", client_id="42")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f4a1"
    job_uuid = "018f7f65-a60f-7c21-b690-0bca9205f4a2"
    jobs_key = _insert_generation_receipt(
        db,
        receipt_id=receipt_id,
        status="queued",
        job_id=7,
        job_uuid=job_uuid,
    )
    jobs = _JobsStore(error=RuntimeError("private Jobs outage"))
    expired_at = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)

    first = reconcile_owner_generation_receipts(
        db,
        jobs,
        owner_user_id="42",
        now=expired_at,
        after_receipt_id=None,
        limit=100,
    )
    jobs.error = None
    jobs.job = _job(
        receipt_id=receipt_id,
        jobs_key=jobs_key,
        status="queued",
        job_uuid=job_uuid,
    )
    second = reconcile_owner_generation_receipts(
        db,
        jobs,
        owner_user_id="42",
        now=expired_at + timedelta(minutes=1),
        after_receipt_id=None,
        limit=100,
    )

    assert first.jobs_available is False
    assert second.jobs_available is True
    assert len(jobs.lookups) == 2
    assert len(jobs.terminalizations) == 1
    assert jobs.terminalizations[0]["error_code"] == "generation_expired"
    db.close_connection()


def test_expired_receipt_does_not_advance_after_jobs_terminal_cas_conflict(tmp_path: Path) -> None:
    db = SlidesDatabase(tmp_path / "Slides.db", client_id="42")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f4a3"
    job_uuid = "018f7f65-a60f-7c21-b690-0bca9205f4a4"
    jobs_key = _insert_generation_receipt(
        db,
        receipt_id=receipt_id,
        status="queued",
        job_id=7,
        job_uuid=job_uuid,
    )
    jobs = _JobsStore(
        _job(
            receipt_id=receipt_id,
            jobs_key=jobs_key,
            status="queued",
            job_uuid=job_uuid,
        ),
        terminalization_outcomes=("CONFLICT", "APPLIED"),
    )
    expired_at = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)

    first = reconcile_owner_generation_receipts(
        db,
        jobs,
        owner_user_id="42",
        now=expired_at,
        after_receipt_id=None,
        limit=100,
    )
    second = reconcile_owner_generation_receipts(
        db,
        jobs,
        owner_user_id="42",
        now=expired_at + timedelta(minutes=1),
        after_receipt_id=None,
        limit=100,
    )

    assert first.jobs_available is False
    assert second.jobs_available is True
    assert [call["error_code"] for call in jobs.terminalizations] == [
        "generation_expired",
        "generation_expired",
    ]
    db.close_connection()


def test_expired_bound_receipt_rejects_a_conflicting_jobs_uuid(tmp_path: Path) -> None:
    db = SlidesDatabase(tmp_path / "Slides.db", client_id="42")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f472"
    expected_job_uuid = "018f7f65-a60f-7c21-b690-0bca9205f473"
    conflicting_job_uuid = "018f7f65-a60f-7c21-b690-0bca9205f474"
    jobs_key = _insert_generation_receipt(
        db,
        receipt_id=receipt_id,
        status="queued",
        job_id=7,
        job_uuid=expected_job_uuid,
    )
    conflicting_job = _job(
        receipt_id=receipt_id,
        jobs_key=jobs_key,
        status="queued",
        job_uuid=conflicting_job_uuid,
    )

    class UuidFilteringJobsStore(_JobsStore):
        def lookup_slides_generation_job(self, **kwargs):
            self.lookups.append(dict(kwargs))
            return None if "expected_job_uuid" in kwargs else self.job

    jobs = UuidFilteringJobsStore(conflicting_job)

    result = reconcile_owner_generation_receipts(
        db,
        jobs,
        owner_user_id="42",
        now=datetime(2026, 7, 20, 12, 0, tzinfo=UTC),
        after_receipt_id=None,
        limit=100,
    )

    receipt = db.get_generation_receipt(receipt_id, owner_user_id="42")
    assert result.jobs_available is False
    assert (receipt.receipt_status, receipt.error_code) == (
        "failed",
        "generation_expired",
    )
    with pytest.raises(KeyError, match="generation_input_not_found"):
        db.get_generation_input(receipt_id, owner_user_id="42")
    assert jobs.terminalizations == []
    assert jobs.lookups == [
        {
            "owner_user_id": "42",
            "idempotency_key": jobs_key,
            "expected_job_uuid": expected_job_uuid,
        },
        {
            "owner_user_id": "42",
            "idempotency_key": jobs_key,
        },
    ]
    db.close_connection()


def test_owner_reconciliation_repairs_an_active_uuid_without_loading_input(tmp_path: Path) -> None:
    db = SlidesDatabase(tmp_path / "Slides.db", client_id="42")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f451"
    job_uuid = "018f7f65-a60f-7c21-b690-0bca9205f452"
    jobs_key = _insert_generation_receipt(db, receipt_id=receipt_id)
    jobs = _JobsStore(
        _job(
            receipt_id=receipt_id,
            jobs_key=jobs_key,
            status="queued",
            job_uuid=job_uuid,
        )
    )

    reconcile_owner_generation_receipts(
        db,
        jobs,
        owner_user_id="42",
        now=datetime(2026, 7, 18, 12, 5, tzinfo=UTC),
        after_receipt_id=None,
        limit=100,
    )

    receipt = db.get_generation_receipt(receipt_id, owner_user_id="42")
    assert (receipt.job_id, receipt.job_uuid, receipt.receipt_status) == (7, job_uuid, "queued")
    assert jobs.lookups == [
        {
            "owner_user_id": "42",
            "idempotency_key": jobs_key,
        }
    ]
    db.close_connection()


def test_owner_reconciliation_repairs_archived_uuid_then_maps_terminal_state(tmp_path: Path) -> None:
    db = SlidesDatabase(tmp_path / "Slides.db", client_id="42")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f453"
    job_uuid = "018f7f65-a60f-7c21-b690-0bca9205f454"
    jobs_key = _insert_generation_receipt(db, receipt_id=receipt_id)
    jobs = _JobsStore(
        _job(
            receipt_id=receipt_id,
            jobs_key=jobs_key,
            status="failed",
            job_uuid=job_uuid,
            job_id=None,
            archived=True,
        )
    )

    reconcile_owner_generation_receipts(
        db,
        jobs,
        owner_user_id="42",
        now=datetime(2026, 7, 18, 12, 5, tzinfo=UTC),
        after_receipt_id=None,
        limit=100,
    )

    receipt = db.get_generation_receipt(receipt_id, owner_user_id="42")
    assert (receipt.job_id, receipt.job_uuid) == (None, job_uuid)
    assert (receipt.receipt_status, receipt.error_code) == ("failed", "generation_provider_failed")
    assert receipt.expires_at == "2026-08-17T12:05:00+00:00"
    db.close_connection()


@pytest.mark.parametrize(
    ("archived", "job_id", "receipt_id", "job_uuid"),
    [
        (
            False,
            7,
            "018f7f65-a60f-7c21-b690-0bca9205f46b",
            "018f7f65-a60f-7c21-b690-0bca9205f46c",
        ),
        (
            True,
            None,
            "018f7f65-a60f-7c21-b690-0bca9205f46d",
            "018f7f65-a60f-7c21-b690-0bca9205f46e",
        ),
    ],
)
def test_owner_reconciliation_uses_the_fixed_quarantine_mapping(
    tmp_path: Path,
    archived: bool,
    job_id: int | None,
    receipt_id: str,
    job_uuid: str,
) -> None:
    db = SlidesDatabase(tmp_path / "Slides.db", client_id="42")
    jobs_key = _insert_generation_receipt(db, receipt_id=receipt_id)
    job = _job(
        receipt_id=receipt_id,
        jobs_key=jobs_key,
        status="quarantined",
        job_uuid=job_uuid,
        job_id=job_id,
        archived=archived,
    )
    job["error_code"] = "generation_provider_failed"
    jobs = _JobsStore(job)

    reconcile_owner_generation_receipts(
        db,
        jobs,
        owner_user_id="42",
        now=datetime(2026, 7, 18, 12, 5, tzinfo=UTC),
        after_receipt_id=None,
        limit=100,
    )

    receipt = db.get_generation_receipt(receipt_id, owner_user_id="42")
    assert (receipt.receipt_status, receipt.error_code) == (
        "failed",
        "generation_quarantined",
    )
    db.close_connection()


def test_bound_receipt_lookup_uses_the_immutable_expected_jobs_uuid(tmp_path: Path) -> None:
    db = SlidesDatabase(tmp_path / "Slides.db", client_id="42")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f4a8"
    job_uuid = "018f7f65-a60f-7c21-b690-0bca9205f4a9"
    jobs_key = _insert_generation_receipt(
        db,
        receipt_id=receipt_id,
        status="queued",
        job_id=7,
        job_uuid=job_uuid,
    )
    jobs = _JobsStore(
        _job(
            receipt_id=receipt_id,
            jobs_key=jobs_key,
            status="queued",
            job_uuid=job_uuid,
        )
    )

    reconcile_owner_generation_receipts(
        db,
        jobs,
        owner_user_id="42",
        now=datetime(2026, 7, 18, 12, 5, tzinfo=UTC),
        after_receipt_id=None,
        limit=100,
    )

    assert jobs.lookups == [
        {
            "owner_user_id": "42",
            "idempotency_key": jobs_key,
            "expected_job_uuid": job_uuid,
        }
    ]
    db.close_connection()


def test_bound_receipt_distinguishes_a_conflicting_jobs_uuid_from_a_miss(tmp_path: Path) -> None:
    db = SlidesDatabase(tmp_path / "Slides.db", client_id="42")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f46f"
    expected_job_uuid = "018f7f65-a60f-7c21-b690-0bca9205f470"
    conflicting_job_uuid = "018f7f65-a60f-7c21-b690-0bca9205f471"
    jobs_key = _insert_generation_receipt(
        db,
        receipt_id=receipt_id,
        status="queued",
        job_id=7,
        job_uuid=expected_job_uuid,
    )
    conflicting_job = _job(
        receipt_id=receipt_id,
        jobs_key=jobs_key,
        status="queued",
        job_uuid=conflicting_job_uuid,
    )

    class UuidFilteringJobsStore(_JobsStore):
        def lookup_slides_generation_job(self, **kwargs):
            self.lookups.append(dict(kwargs))
            return None if "expected_job_uuid" in kwargs else self.job

    jobs = UuidFilteringJobsStore(conflicting_job)

    reconcile_owner_generation_receipts(
        db,
        jobs,
        owner_user_id="42",
        now=datetime(2026, 7, 18, 12, 5, tzinfo=UTC),
        after_receipt_id=None,
        limit=100,
    )

    receipt = db.get_generation_receipt(receipt_id, owner_user_id="42")
    assert (receipt.receipt_status, receipt.error_code) == (
        "failed",
        "generation_correlation_mismatch",
    )
    assert jobs.lookups == [
        {
            "owner_user_id": "42",
            "idempotency_key": jobs_key,
            "expected_job_uuid": expected_job_uuid,
        },
        {
            "owner_user_id": "42",
            "idempotency_key": jobs_key,
        },
    ]
    db.close_connection()


def test_owner_reconciliation_requires_fifteen_minutes_of_confirmed_misses(tmp_path: Path) -> None:
    db = SlidesDatabase(tmp_path / "Slides.db", client_id="42")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f455"
    _insert_generation_receipt(db, receipt_id=receipt_id)
    jobs = _JobsStore(job=None)
    first = datetime(2026, 7, 18, 12, 5, tzinfo=UTC)

    for observed_at in (first, first + timedelta(seconds=899)):
        reconcile_owner_generation_receipts(
            db,
            jobs,
            owner_user_id="42",
            now=observed_at,
            after_receipt_id=None,
            limit=100,
        )
        pending = db.get_generation_receipt(receipt_id, owner_user_id="42")
        assert pending.receipt_status == "claimed"
        assert pending.error_code == "generation_receipt_unresolved_pending"
        assert pending.updated_at == "2026-07-18T12:05:00+00:00"

    reconcile_owner_generation_receipts(
        db,
        jobs,
        owner_user_id="42",
        now=first + timedelta(seconds=900),
        after_receipt_id=None,
        limit=100,
    )
    terminal = db.get_generation_receipt(receipt_id, owner_user_id="42")
    assert (terminal.receipt_status, terminal.error_code) == (
        "failed",
        "generation_receipt_unresolved",
    )
    assert terminal.updated_at == "2026-07-18T12:20:00+00:00"
    db.close_connection()


def test_jobs_outage_neither_starts_nor_advances_missing_confirmation(tmp_path: Path) -> None:
    db = SlidesDatabase(tmp_path / "Slides.db", client_id="42")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f456"
    _insert_generation_receipt(db, receipt_id=receipt_id)
    first = datetime(2026, 7, 18, 12, 5, tzinfo=UTC)
    reconcile_owner_generation_receipts(
        db,
        _JobsStore(job=None),
        owner_user_id="42",
        now=first,
        after_receipt_id=None,
        limit=100,
    )

    result = reconcile_owner_generation_receipts(
        db,
        _JobsStore(error=RuntimeError("sensitive Jobs failure")),
        owner_user_id="42",
        now=first + timedelta(hours=1),
        after_receipt_id=None,
        limit=100,
    )

    receipt = db.get_generation_receipt(receipt_id, owner_user_id="42")
    assert result.jobs_available is False
    assert receipt.receipt_status == "claimed"
    assert receipt.error_code == "generation_receipt_unresolved_pending"
    assert receipt.updated_at == "2026-07-18T12:05:00+00:00"
    db.close_connection()


def test_matching_completed_presentation_wins_over_cancelled_jobs_state(tmp_path: Path) -> None:
    db = SlidesDatabase(tmp_path / "Slides.db", client_id="42")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f457"
    job_uuid = "018f7f65-a60f-7c21-b690-0bca9205f458"
    db.create_presentation(
        presentation_id=receipt_id,
        title="Committed",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        slides="[]",
        slides_text="safe text",
        source_type="prompt",
        source_ref=None,
        source_query=None,
        custom_css=None,
        content_kind="standalone_html",
        html_document="<html><body><section>secret</section></body></html>",
        html_sha256=_sha256("<html><body><section>secret</section></body></html>"),
        html_bytes=len("<html><body><section>secret</section></body></html>"),
        html_slide_count=1,
        generation_job_uuid=job_uuid,
        generation_provenance_json='{"source":{"kind":"prompt"}}',
    )
    jobs_key = _insert_generation_receipt(
        db,
        receipt_id=receipt_id,
        status="completed",
        job_id=7,
        job_uuid=job_uuid,
        presentation_id=receipt_id,
        include_input=True,
    )
    jobs = _JobsStore(
        _job(
            receipt_id=receipt_id,
            jobs_key=jobs_key,
            status="cancelled",
            job_uuid=job_uuid,
        )
    )

    reconcile_owner_generation_receipts(
        db,
        jobs,
        owner_user_id="42",
        now=datetime(2026, 7, 18, 12, 5, tzinfo=UTC),
        after_receipt_id=None,
        limit=100,
    )

    receipt = db.get_generation_receipt(receipt_id, owner_user_id="42")
    assert receipt.receipt_status == "completed"
    assert receipt.presentation_id == receipt_id
    with pytest.raises(KeyError, match="generation_input_not_found"):
        db.get_generation_input(receipt_id, owner_user_id="42")
    assert jobs.lookups == []
    db.close_connection()


def test_completed_receipt_without_matching_presentation_fails_sweep_closed(tmp_path: Path) -> None:
    db = SlidesDatabase(tmp_path / "Slides.db", client_id="42")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f4a5"
    job_uuid = "018f7f65-a60f-7c21-b690-0bca9205f4a6"
    html = "<html><body><section>safe</section></body></html>"
    db.create_presentation(
        presentation_id=receipt_id,
        title="Mismatched commit",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        slides="[]",
        slides_text="safe",
        source_type="prompt",
        source_ref=None,
        source_query=None,
        custom_css=None,
        content_kind="standalone_html",
        html_document=html,
        html_sha256=_sha256(html),
        html_bytes=len(html.encode("utf-8")),
        html_slide_count=1,
        generation_job_uuid="018f7f65-a60f-7c21-b690-0bca9205f4a7",
        generation_provenance_json='{"source":{"kind":"prompt"}}',
    )
    _insert_generation_receipt(
        db,
        receipt_id=receipt_id,
        status="completed",
        job_id=7,
        job_uuid=job_uuid,
        presentation_id=receipt_id,
        include_input=True,
    )
    jobs = _JobsStore()

    result = reconcile_owner_generation_receipts(
        db,
        jobs,
        owner_user_id="42",
        now=datetime(2026, 7, 18, 12, 5, tzinfo=UTC),
        after_receipt_id=None,
        limit=100,
    )

    receipt = db.get_generation_receipt(receipt_id, owner_user_id="42")
    assert result.jobs_available is False
    assert receipt.receipt_status == "completed"
    with pytest.raises(KeyError, match="generation_input_not_found"):
        db.get_generation_input(receipt_id, owner_user_id="42")
    assert jobs.lookups == []
    db.close_connection()


def test_completed_jobs_without_a_matching_presentation_fails_correlation(tmp_path: Path) -> None:
    db = SlidesDatabase(tmp_path / "Slides.db", client_id="42")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f459"
    job_uuid = "018f7f65-a60f-7c21-b690-0bca9205f45a"
    jobs_key = _insert_generation_receipt(
        db,
        receipt_id=receipt_id,
        status="queued",
        job_id=7,
        job_uuid=job_uuid,
    )
    jobs = _JobsStore(
        _job(
            receipt_id=receipt_id,
            jobs_key=jobs_key,
            status="completed",
            job_uuid=job_uuid,
        )
    )

    reconcile_owner_generation_receipts(
        db,
        jobs,
        owner_user_id="42",
        now=datetime(2026, 7, 18, 12, 5, tzinfo=UTC),
        after_receipt_id=None,
        limit=100,
    )

    receipt = db.get_generation_receipt(receipt_id, owner_user_id="42")
    assert (receipt.receipt_status, receipt.error_code) == (
        "failed",
        "generation_correlation_mismatch",
    )
    db.close_connection()


class _CoordinationStore:
    def __init__(
        self,
        *,
        now: datetime,
        active_owners: tuple[str, ...] = (),
        retiring_key_id: str | None = None,
    ) -> None:
        self.now = now
        self.active_owners = active_owners
        records: list[dict[str, object]] = [
            {
                "key_id": "key-v2",
                "state": "current",
                "activated_at": now - timedelta(days=40),
                "retired_at": None,
                "config_revision": "epoch-1",
            }
        ]
        if retiring_key_id is not None:
            records.append(
                {
                    "key_id": retiring_key_id,
                    "state": "retiring",
                    "activated_at": now - timedelta(days=80),
                    "retired_at": now - timedelta(days=40),
                    "config_revision": "epoch-1",
                }
            )
        self.registry = {"records": records, "config_revision": "epoch-1"}
        self.state: dict[str, object] = {
            "holder_uuid": None,
            "lease_expires_at": None,
            "fencing_token": 0,
            "cursor": None,
            "config_revision": "epoch-1",
            "startup_complete_epoch": None,
            "last_complete_epoch": None,
            "lag": 0,
            "sweep_key_id": None,
            "sweep_started_at": None,
            "sweep_completed_at": None,
            "sweep_complete": False,
            "unexpired_reference_count": 0,
        }
        self.checkpoints: list[dict[str, object]] = []
        self.job_error: Exception | None = None
        self.coordination_error: Exception | None = None

    def get_slides_generation_readiness(self) -> dict[str, object]:
        if self.coordination_error is not None:
            raise self.coordination_error
        return {"ready": True}

    def get_slides_reconciliation_state(self) -> dict[str, object]:
        return dict(self.state)

    def acquire_slides_reconciliation_lease(self, **kwargs) -> dict[str, object] | None:
        observed_expiry = self.state.get("lease_expires_at")
        if isinstance(observed_expiry, datetime) and observed_expiry > kwargs["now"]:
            return None
        same_revision = self.state.get("config_revision") == kwargs["config_revision"]
        if not same_revision:
            self.state.update(
                cursor=None,
                startup_complete_epoch=None,
                last_complete_epoch=None,
                lag=0,
                config_revision=kwargs["config_revision"],
            )
        self.state.update(
            holder_uuid=kwargs["holder_uuid"],
            lease_expires_at=kwargs["now"] + timedelta(seconds=kwargs["lease_seconds"]),
            fencing_token=int(self.state["fencing_token"]) + 1,
            sweep_key_id=None,
            sweep_started_at=None,
            sweep_completed_at=None,
            sweep_complete=False,
            unexpired_reference_count=0,
        )
        return dict(self.state)

    def renew_slides_reconciliation_lease(self, **kwargs) -> bool:
        if not self._owns(kwargs):
            return False
        self.state["lease_expires_at"] = kwargs["now"] + timedelta(seconds=kwargs["lease_seconds"])
        return True

    def checkpoint_slides_reconciliation(self, **kwargs) -> bool:
        self.checkpoints.append(dict(kwargs))
        if not self._owns(kwargs):
            return False
        self.state.update(
            cursor=None if kwargs["completed"] else kwargs["cursor"],
            startup_complete_epoch=kwargs["startup_complete_epoch"],
            last_complete_epoch=(
                kwargs["now"].timestamp()
                if kwargs["completed"] and kwargs["last_complete_epoch"] is None
                else kwargs["last_complete_epoch"]
            ),
            lag=0 if kwargs["completed"] else kwargs["lag"],
        )
        if kwargs.get("sweep_key_id") is not None:
            self.state.update(
                sweep_key_id=kwargs["sweep_key_id"],
                sweep_started_at=kwargs["sweep_started_at"],
                sweep_completed_at=kwargs["now"] if kwargs["completed"] else None,
                sweep_complete=bool(kwargs["completed"]),
                unexpired_reference_count=kwargs["unexpired_reference_count"],
            )
        return True

    def release_slides_reconciliation_lease(self, **kwargs) -> bool:
        if not self._owns(kwargs, require_live=False):
            return False
        self.state.update(holder_uuid=None, lease_expires_at=None)
        return True

    def load_slides_digest_key_registry(self) -> dict[str, object]:
        return self.registry

    def list_active_slides_generation_owner_ids(self, **kwargs) -> list[str]:
        after = kwargs["after_owner_user_id"]
        candidates = [owner for owner in sorted(set(self.active_owners)) if after is None or owner > after]
        return candidates[: kwargs["limit"]]

    def lookup_slides_generation_job(self, **kwargs):
        del kwargs
        if self.job_error is not None:
            raise self.job_error
        return None

    def _owns(self, kwargs: Mapping[str, object], *, require_live: bool = True) -> bool:
        expiry = self.state.get("lease_expires_at")
        return bool(
            self.state.get("holder_uuid") == kwargs["holder_uuid"]
            and self.state.get("fencing_token") == kwargs["fencing_token"]
            and self.state.get("config_revision") == kwargs["config_revision"]
            and (not require_live or isinstance(expiry, datetime) and expiry > kwargs["now"])
        )


class _DatabaseOpenTracker:
    def __init__(self) -> None:
        self.open_count = 0
        self.max_open_count = 0
        self.opened_owner_ids: list[str] = []

    def factory(
        self,
        *,
        db_path: str | Path,
        client_id: str,
        expected_file_identity: tuple[int, int, int],
        expected_directory_identities: tuple[
            tuple[str | Path, tuple[int, int, int]],
            ...,
        ],
    ) -> SlidesDatabase:
        tracker = self

        class TrackedSlidesDatabase(SlidesDatabase):
            def close_connection(self) -> None:
                super().close_connection()
                if not self._tracker_closed:
                    self._tracker_closed = True
                    tracker.open_count -= 1

        slides_db = TrackedSlidesDatabase.open_existing_complete(
            db_path=db_path,
            client_id=client_id,
            expected_file_identity=expected_file_identity,
            expected_directory_identities=expected_directory_identities,
        )
        tracker.open_count += 1
        tracker.max_open_count = max(tracker.max_open_count, tracker.open_count)
        tracker.opened_owner_ids.append(client_id)
        slides_db._tracker_closed = False
        return slides_db


def _run_until_complete(
    reconciler: FencedStandaloneHtmlReconciler,
    *,
    limit: int = 30,
):
    results = []
    for _ in range(limit):
        result = reconciler.run_batch()
        results.append(result)
        if result.completed_pass:
            return results
    raise AssertionError("reconciliation pass did not complete")


def test_admission_rechecks_time_after_blocking_shared_state_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    store = _CoordinationStore(now=now)
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=tmp_path / "user_databases",
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
    )
    _run_until_complete(reconciler)

    def delayed_state_read() -> dict[str, object]:
        state = dict(store.state)
        lease_expires_at = state["lease_expires_at"]
        assert isinstance(lease_expires_at, datetime)
        store.now = lease_expires_at
        return state

    monkeypatch.setattr(store, "get_slides_reconciliation_state", delayed_state_read)

    assert reconciler.admission_ready() is False


def test_admission_rejects_a_later_diagnostic_state_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    store = _CoordinationStore(now=now)
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=tmp_path / "user_databases",
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
    )
    _run_until_complete(reconciler)

    def diagnosed_state_read() -> dict[str, object]:
        state = dict(store.state)
        state["diagnostic_code"] = "generation_archive_correlation_failed"
        return state

    monkeypatch.setattr(store, "get_slides_reconciliation_state", diagnosed_state_read)

    assert reconciler.admission_ready() is False


def test_fenced_revalidates_discovered_database_before_factory_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    base_dir = tmp_path / "user_databases"
    db_path = _create_slides_db(base_dir, "2")
    store = _CoordinationStore(now=now, active_owners=("2",))
    original_resolve = reconciler_module._resolve_canonical_owner_database
    opened_paths: list[Path] = []

    def disappearing_resolve(*args, **kwargs):
        discovered = original_resolve(*args, **kwargs)
        assert discovered is not None
        discovered.path.unlink()
        return discovered

    def tracking_factory(
        *,
        db_path: str | Path,
        client_id: str,
        expected_file_identity: tuple[int, int, int],
        expected_directory_identities: tuple[
            tuple[str | Path, tuple[int, int, int]],
            ...,
        ],
    ) -> SlidesDatabase:
        del expected_file_identity, expected_directory_identities
        opened_paths.append(Path(db_path))
        return SlidesDatabase(db_path=db_path, client_id=client_id)

    monkeypatch.setattr(
        reconciler_module,
        "_resolve_canonical_owner_database",
        disappearing_resolve,
    )
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
        slides_db_factory=tracking_factory,
    )

    result = reconciler.run_batch()

    assert result.diagnostic_code == "standalone_html_slides_database_unsafe"
    assert opened_paths == []
    assert not db_path.exists()


def test_fenced_open_does_not_recreate_a_database_removed_after_final_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    base_dir = tmp_path / "user_databases"
    db_path = _create_slides_db(base_dir, "2")
    store = _CoordinationStore(now=now, active_owners=("2",))
    original_revalidate = reconciler_module._revalidate_discovered_database

    def remove_after_revalidation(discovered):
        revalidated = original_revalidate(discovered)
        db_path.unlink()
        return revalidated

    monkeypatch.setattr(
        reconciler_module,
        "_revalidate_discovered_database",
        remove_after_revalidation,
    )
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
    )

    result = reconciler.run_batch()

    assert result.diagnostic_code == "standalone_html_slides_database_unsafe"
    assert not db_path.exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_fenced_open_rejects_a_symlink_inserted_after_final_revalidation_without_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    base_dir = tmp_path / "user_databases"
    db_path = _create_slides_db(base_dir, "2")
    outside_target = tmp_path / "outside.db"
    outside_target.touch()
    store = _CoordinationStore(now=now, active_owners=("2",))
    original_revalidate = reconciler_module._revalidate_discovered_database

    def replace_with_symlink_after_revalidation(discovered):
        revalidated = original_revalidate(discovered)
        db_path.unlink()
        db_path.symlink_to(outside_target)
        return revalidated

    monkeypatch.setattr(
        reconciler_module,
        "_revalidate_discovered_database",
        replace_with_symlink_after_revalidation,
    )
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
    )

    result = reconciler.run_batch()

    assert result.diagnostic_code == "standalone_html_slides_database_unsafe"
    assert outside_target.stat().st_size == 0


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_fenced_open_rejects_an_owner_directory_symlink_inserted_after_final_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    base_dir = tmp_path / "user_databases"
    db_path = _create_slides_db(base_dir, "2")
    moved_owner_directory = tmp_path / "moved-owner"
    store = _CoordinationStore(now=now, active_owners=("2",))
    original_revalidate = reconciler_module._revalidate_discovered_database

    def replace_owner_directory_with_symlink(discovered):
        revalidated = original_revalidate(discovered)
        os.replace(db_path.parent, moved_owner_directory)
        db_path.parent.symlink_to(moved_owner_directory, target_is_directory=True)
        return revalidated

    monkeypatch.setattr(
        reconciler_module,
        "_revalidate_discovered_database",
        replace_owner_directory_with_symlink,
    )
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
    )

    result = reconciler.run_batch()

    assert result.diagnostic_code == "standalone_html_slides_database_unsafe"


def test_fenced_open_rejects_an_inode_replaced_after_final_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    base_dir = tmp_path / "user_databases"
    db_path = _create_slides_db(base_dir, "2")
    replacement_path = _create_slides_db(tmp_path / "replacement_registry", "99")
    store = _CoordinationStore(now=now, active_owners=("2",))
    original_revalidate = reconciler_module._revalidate_discovered_database

    def replace_inode_after_revalidation(discovered):
        revalidated = original_revalidate(discovered)
        os.replace(replacement_path, db_path)
        return revalidated

    monkeypatch.setattr(
        reconciler_module,
        "_revalidate_discovered_database",
        replace_inode_after_revalidation,
    )
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
    )

    result = reconciler.run_batch()

    assert result.diagnostic_code == "standalone_html_slides_database_unsafe"


def test_local_expiry_does_not_recreate_a_database_removed_after_final_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    base_dir = tmp_path / "user_databases"
    db_path = _create_slides_db(base_dir, "2")
    store = _CoordinationStore(now=now)
    original_revalidate = reconciler_module._revalidate_discovered_database

    def remove_after_revalidation(discovered):
        revalidated = original_revalidate(discovered)
        db_path.unlink()
        return revalidated

    monkeypatch.setattr(
        reconciler_module,
        "_revalidate_discovered_database",
        remove_after_revalidation,
    )
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
    )

    result = reconciler.run_local_expiry_batch()

    assert result.diagnostic_code == "standalone_html_slides_database_unsafe"
    assert not db_path.exists()


@pytest.mark.parametrize("malformed_state", ["cursor", "registry"])
def test_acquisition_releases_lease_when_shared_state_is_malformed(
    tmp_path: Path,
    malformed_state: str,
) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    store = _CoordinationStore(now=now)
    if malformed_state == "cursor":
        store.state["cursor"] = "not-json"
    else:
        store.registry["records"] = "not-a-record-list"
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=tmp_path / "missing_user_databases",
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
    )

    result = reconciler.run_batch()

    assert result.jobs_available is False
    assert store.state["holder_uuid"] is None
    assert store.state["lease_expires_at"] is None


def test_fenced_sweep_prioritizes_active_owner_closes_each_db_and_publishes_startup(tmp_path: Path) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    base_dir = tmp_path / "user_databases"
    _create_slides_db(base_dir, "2")
    _create_slides_db(base_dir, "10")
    store = _CoordinationStore(now=now, active_owners=("10",))
    tracker = _DatabaseOpenTracker()
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
        slides_db_factory=tracker.factory,
    )

    results = _run_until_complete(reconciler)

    assert tracker.opened_owner_ids == ["10", "2", "10"]
    assert tracker.max_open_count == 1
    assert tracker.open_count == 0
    assert store.state["startup_complete_epoch"] == "epoch-1"
    assert store.state["cursor"] is None
    assert results[-1].startup_ready is True


def test_jobs_outage_applies_absolute_expiry_but_withholds_fenced_progress(tmp_path: Path) -> None:
    now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    base_dir = tmp_path / "user_databases"
    db_path = _create_slides_db(base_dir, "2")
    db = SlidesDatabase(db_path, client_id="2")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f45b"
    _insert_generation_receipt(
        db,
        receipt_id=receipt_id,
        owner_user_id="2",
        status="queued",
        job_id=7,
        job_uuid="018f7f65-a60f-7c21-b690-0bca9205f45c",
    )
    db.close_connection()
    store = _CoordinationStore(now=now, active_owners=("2",))
    store.job_error = RuntimeError("private Jobs outage")
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
    )

    result = reconciler.run_batch()

    checked = SlidesDatabase(db_path, client_id="2")
    receipt = checked.get_generation_receipt(receipt_id, owner_user_id="2")
    assert (receipt.receipt_status, receipt.updated_at) == (
        "failed",
        "2026-07-19T12:00:00+00:00",
    )
    checked.close_connection()
    assert result.jobs_available is False
    assert store.checkpoints == []
    assert store.state["cursor"] is None


def test_lookup_outage_advances_the_independent_local_expiry_sweep(tmp_path: Path) -> None:
    now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    base_dir = tmp_path / "user_databases"
    active_path = _create_slides_db(base_dir, "2")
    dormant_path = _create_slides_db(base_dir, "10")
    active_db = SlidesDatabase(active_path, client_id="2")
    dormant_db = SlidesDatabase(dormant_path, client_id="10")
    active_id = "018f7f65-a60f-7c21-b690-0bca9205f45f"
    dormant_id = "018f7f65-a60f-7c21-b690-0bca9205f460"
    _insert_generation_receipt(
        active_db,
        receipt_id=active_id,
        owner_user_id="2",
        status="queued",
        input_expires_at="2026-07-21T00:00:00+00:00",
    )
    with active_db.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE slides_generation_receipts SET created_at = ? WHERE id = ?",
            ("2026-07-20T00:00:00+00:00", active_id),
        )
    _insert_generation_receipt(
        dormant_db,
        receipt_id=dormant_id,
        owner_user_id="10",
        status="queued",
    )
    active_db.close_connection()
    dormant_db.close_connection()
    store = _CoordinationStore(now=now, active_owners=("2",))
    store.job_error = RuntimeError("private Jobs outage")
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
    )

    first = reconciler.run_batch()
    second = reconciler.run_batch()

    checked = SlidesDatabase(dormant_path, client_id="10")
    receipt = checked.get_generation_receipt(dormant_id, owner_user_id="10")
    checked.close_connection()
    assert first.local_sweep_state == "progressed"
    assert second.local_sweep_state == "progressed"
    assert second.processed_owner_user_id == "10"
    assert (receipt.receipt_status, receipt.error_code) == (
        "failed",
        "generation_expired",
    )
    assert store.checkpoints == []


def test_first_pass_checkpoint_reports_elapsed_overload_lag(tmp_path: Path) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    base_dir = tmp_path / "user_databases"
    _create_slides_db(base_dir, "2")
    store = _CoordinationStore(now=now, active_owners=("2",))
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
        lease_seconds=1800,
    )

    reconciler.run_batch()
    store.now += timedelta(seconds=901)
    reconciler.run_batch()

    assert store.checkpoints[-1]["lag"] == 901


def test_owner_page_cannot_checkpoint_after_lease_expires_during_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    base_dir = tmp_path / "user_databases"
    db_path = _create_slides_db(base_dir, "2")
    db = SlidesDatabase(db_path, client_id="2")
    _insert_generation_receipt(
        db,
        receipt_id="018f7f65-a60f-7c21-b690-0bca9205f462",
        owner_user_id="2",
        status="queued",
    )
    db.close_connection()
    store = _CoordinationStore(now=now, active_owners=("2",))

    def expire_lease_during_lookup(**_kwargs):
        store.now += timedelta(seconds=31)
        return None

    monkeypatch.setattr(
        store,
        "lookup_slides_generation_job",
        expire_lease_during_lookup,
    )
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
        lease_seconds=30,
    )

    result = reconciler.run_batch()

    assert result.lost_leadership is True
    assert result.startup_ready is False
    assert store.state["cursor"] is None
    assert store.state["startup_complete_epoch"] is None
    assert store.checkpoints[-1]["now"] == store.now


def test_local_expiry_terminalizes_corrupt_authoritative_created_at(tmp_path: Path) -> None:
    db = SlidesDatabase(tmp_path / "Slides.db", client_id="42")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f461"
    _insert_generation_receipt(
        db,
        receipt_id=receipt_id,
        status="queued",
    )
    with db.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE slides_generation_receipts SET created_at = ? WHERE id = ?",
            ("not-a-canonical-timestamp", receipt_id),
        )

    reconcile_owner_local_expiry(
        db,
        owner_user_id="42",
        now=datetime(2026, 7, 20, 12, 0, tzinfo=UTC),
        after_receipt_id=None,
        limit=100,
    )

    receipt = db.get_generation_receipt(receipt_id, owner_user_id="42")
    assert (receipt.receipt_status, receipt.error_code) == (
        "failed",
        "generation_correlation_mismatch",
    )
    with pytest.raises(KeyError, match="generation_input_not_found"):
        db.get_generation_input(receipt_id, owner_user_id="42")
    db.close_connection()


def test_coordination_outage_runs_unfenced_local_expiry_only_without_publication(tmp_path: Path) -> None:
    now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    base_dir = tmp_path / "user_databases"
    db_path = _create_slides_db(base_dir, "2")
    db = SlidesDatabase(db_path, client_id="2")
    expired_id = "018f7f65-a60f-7c21-b690-0bca9205f45d"
    live_id = "018f7f65-a60f-7c21-b690-0bca9205f45e"
    _insert_generation_receipt(
        db,
        receipt_id=expired_id,
        owner_user_id="2",
        status="queued",
        input_expires_at="2026-07-19T12:00:00+00:00",
    )
    _insert_generation_receipt(
        db,
        receipt_id=live_id,
        owner_user_id="2",
        status="claimed",
        input_expires_at="2026-07-21T12:00:00+00:00",
    )
    with db.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE slides_generation_receipts SET created_at = ? WHERE id = ?",
            ("2026-07-20T12:00:00+00:00", live_id),
        )
    db.close_connection()
    store = _CoordinationStore(now=now)
    store.coordination_error = RuntimeError("private coordination outage")
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
    )

    result = reconciler.run_batch()

    checked = SlidesDatabase(db_path, client_id="2")
    expired = checked.get_generation_receipt(expired_id, owner_user_id="2")
    live = checked.get_generation_receipt(live_id, owner_user_id="2")
    assert (expired.receipt_status, expired.error_code) == ("failed", "generation_expired")
    assert (live.receipt_status, live.error_code) == ("claimed", None)
    checked.close_connection()
    assert result.jobs_available is False
    assert result.processed_owner_user_id == "2"
    assert store.checkpoints == []


def test_takeover_resumes_cleanup_but_requires_fresh_pass_for_retiring_key_proof(tmp_path: Path) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    base_dir = tmp_path / "user_databases"
    _create_slides_db(base_dir, "2")
    store = _CoordinationStore(now=now, retiring_key_id="key-v1")
    first = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
        lease_seconds=30,
    )

    first.run_batch()
    first.run_batch()
    assert store.state["cursor"] is not None
    store.now += timedelta(seconds=31)
    second = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-b",
        now=lambda: store.now,
        lease_seconds=30,
    )

    resumed = _run_until_complete(second)

    assert resumed[-1].startup_ready is False
    assert store.state["startup_complete_epoch"] is None
    assert store.state["sweep_complete"] is False

    fresh = _run_until_complete(second)

    assert fresh[-1].startup_ready is True
    assert store.state["sweep_key_id"] == "key-v1"
    assert store.state["sweep_complete"] is True
    assert store.state["unexpired_reference_count"] == 0


def test_mismatched_database_owner_blocks_retiring_key_proof(tmp_path: Path) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    base_dir = tmp_path / "user_databases"
    db_path = _create_slides_db(base_dir, "2")
    db = SlidesDatabase(db_path, client_id="2")
    _insert_generation_receipt(
        db,
        receipt_id="018f7f65-a60f-7c21-b690-0bca9205f463",
        owner_user_id="3",
        status="queued",
    )
    db.close_connection()
    store = _CoordinationStore(now=now, retiring_key_id="key-v1")
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
    )

    reconciler.run_batch()
    checkpoint_count = len(store.checkpoints)
    result = reconciler.run_batch()

    assert result.startup_ready is False
    assert result.jobs_available is False
    assert result.local_sweep_state == "blocked"
    assert len(store.checkpoints) == checkpoint_count
    assert store.state["sweep_complete"] is False
    assert store.state["startup_complete_epoch"] is None


def test_local_expiry_sweep_blocks_on_mismatched_database_owner(tmp_path: Path) -> None:
    now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    base_dir = tmp_path / "user_databases"
    db_path = _create_slides_db(base_dir, "2")
    db = SlidesDatabase(db_path, client_id="2")
    receipt_id = "018f7f65-a60f-7c21-b690-0bca9205f464"
    _insert_generation_receipt(
        db,
        receipt_id=receipt_id,
        owner_user_id="3",
        status="queued",
    )
    db.close_connection()
    reconciler = FencedStandaloneHtmlReconciler(
        job_manager=object(),
        user_db_base_dir=base_dir,
        config_epoch="local-only",
        holder_uuid="local-cleaner",
        now=lambda: now,
    )

    result = reconciler.run_local_expiry_batch()

    assert result.local_sweep_state == "blocked"
    checked = SlidesDatabase(db_path, client_id="2")
    assert checked.get_generation_input(receipt_id, owner_user_id="3").receipt_id == receipt_id
    checked.close_connection()


def test_stale_fence_cannot_publish_after_takeover(tmp_path: Path) -> None:
    now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    base_dir = tmp_path / "user_databases"
    _create_slides_db(base_dir, "2")
    store = _CoordinationStore(now=now)
    stale = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-a",
        now=lambda: store.now,
        lease_seconds=30,
    )
    stale.run_batch()
    store.now += timedelta(seconds=31)
    winner = FencedStandaloneHtmlReconciler(
        job_manager=store,
        user_db_base_dir=base_dir,
        config_epoch="epoch-1",
        holder_uuid="leader-b",
        now=lambda: store.now,
        lease_seconds=30,
    )
    winner.run_batch()
    checkpoint_count = len(store.checkpoints)

    result = stale.run_batch()

    assert result.lost_leadership is True
    assert result.startup_ready is False
    assert len(store.checkpoints) == checkpoint_count
