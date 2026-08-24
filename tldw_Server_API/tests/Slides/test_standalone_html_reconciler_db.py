"""Source-free Slides persistence contracts used by generation reconciliation."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.Slides.slides_db import (
    InputError,
    SlidesDatabase,
)

_OWNER = "owner-1"
_CREATED_AT = "2026-07-18T12:00:00+00:00"
_INPUT_EXPIRES_AT = "2026-07-19T12:00:00+00:00"


@pytest.fixture
def slides_db(tmp_path):
    database = SlidesDatabase(tmp_path / "Slides.db", client_id="reconciler-test")
    try:
        yield database
    finally:
        database.close_connection()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _insert_receipt(
    database: SlidesDatabase,
    receipt_id: str,
    *,
    owner_user_id: str = _OWNER,
    digest_key_id: str = "key-v1",
    receipt_status: str = "claimed",
    job_id: int | None = None,
    job_uuid: str | None = None,
    presentation_id: str | None = None,
    error_code: str | None = None,
    created_at: str = _CREATED_AT,
    updated_at: str = _CREATED_AT,
    expires_at: str | None = None,
    input_expires_at: str = _INPUT_EXPIRES_AT,
    input_created_at: str = _CREATED_AT,
    include_input: bool = True,
) -> None:
    with database.transaction(immediate=True) as connection:
        connection.execute(
            """
            INSERT INTO slides_generation_receipts (
                id, owner_user_id, digest_key_id,
                idempotency_key_hmac_sha256, jobs_idempotency_key,
                client_request_hmac_sha256, execution_hmac_sha256,
                job_id, job_uuid, presentation_id, receipt_status,
                error_code, error_message, created_at, updated_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                owner_user_id,
                digest_key_id,
                _digest(f"idempotency:{receipt_id}"),
                f"slides:v1:{_digest(f'jobs:{receipt_id}')}",
                _digest(f"request:{receipt_id}"),
                _digest(f"execution:{receipt_id}"),
                job_id,
                job_uuid,
                presentation_id,
                receipt_status,
                error_code,
                "safe retry metadata" if error_code else None,
                created_at,
                updated_at,
                expires_at,
            ),
        )
        if include_input:
            source_text = f"SOURCE-SECRET:{receipt_id}"
            system_prompt = f"PROMPT-SECRET:{receipt_id}"
            connection.execute(
                """
                INSERT INTO slides_generation_inputs (
                    receipt_id, source_kind, source_text, source_hmac_sha256,
                    source_bytes, provenance_json, html_options_json, provider,
                    model, adapter_id, endpoint_identity, system_prompt,
                    prompt_sha256, prompt_contract_version, input_expires_at,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    "prompt",
                    source_text,
                    _digest(source_text),
                    len(source_text.encode("utf-8")),
                    f'{{"secret":"PROVENANCE-SECRET:{receipt_id}"}}',
                    "{}",
                    "openai",
                    "gpt-test",
                    "openai_official_chat_v1",
                    "https://api.openai.com:443/v1/chat/completions",
                    system_prompt,
                    _digest(system_prompt),
                    "slides.standalone_html.v1",
                    input_expires_at,
                    input_created_at,
                ),
            )


def _create_standalone_presentation(
    database: SlidesDatabase,
    presentation_id: str,
    *,
    job_uuid: str,
) -> None:
    html_document = f"<html><body><section>HTML-SECRET:{presentation_id}</section></body></html>"
    database.create_presentation(
        presentation_id=presentation_id,
        title="Reconciled presentation",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        slides="[]",
        slides_text="safe derived text",
        source_type="prompt",
        source_ref=None,
        source_query=None,
        custom_css=None,
        content_kind="standalone_html",
        html_document=html_document,
        html_sha256=_digest(html_document),
        html_bytes=len(html_document.encode("utf-8")),
        html_slide_count=1,
        generation_job_uuid=job_uuid,
        generation_provenance_json=f'{{"secret":"PRESENTATION-PROVENANCE:{presentation_id}"}}',
    )


def test_reconciliation_scan_is_bounded_and_source_free(slides_db):
    _insert_receipt(slides_db, "receipt-1")
    _insert_receipt(slides_db, "receipt-2")
    _insert_receipt(slides_db, "receipt-3")
    connection = slides_db.get_connection()
    forbidden = {
        ("slides_generation_inputs", "source_text"),
        ("slides_generation_inputs", "system_prompt"),
        ("slides_generation_inputs", "provenance_json"),
        ("presentations", "html_document"),
        ("presentations", "generation_provenance_json"),
    }

    def deny_source_reads(action, table, column, _database, _trigger):
        if action == sqlite3.SQLITE_READ and (table, column) in forbidden:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(deny_source_reads)
    try:
        rows = slides_db.list_generation_receipts_for_reconciliation(
            owner_user_id=_OWNER,
            after_receipt_id="receipt-1",
            limit=1,
        )
    finally:
        connection.set_authorizer(None)

    assert [row.id for row in rows] == ["receipt-2"]
    assert set(asdict(rows[0])).isdisjoint(
        {
            "source_text",
            "system_prompt",
            "provenance_json",
            "html_document",
            "generation_provenance_json",
        }
    )


def test_reconciliation_scan_exposes_a_physical_database_owner_mismatch(slides_db):
    _insert_receipt(slides_db, "receipt-other", owner_user_id="owner-2")

    rows = slides_db.list_generation_receipts_for_reconciliation(
        owner_user_id=_OWNER,
        after_receipt_id=None,
        limit=10,
    )

    assert [(row.id, row.owner_user_id) for row in rows] == [("receipt-other", "owner-2")]


def test_reconciliation_scan_rejects_an_unbounded_limit(slides_db):
    with pytest.raises(InputError, match="reconciliation limit"):
        slides_db.list_generation_receipts_for_reconciliation(
            owner_user_id=_OWNER,
            after_receipt_id=None,
            limit=501,
        )


def test_reconciliation_scan_reports_matching_presentation_metadata(slides_db):
    job_uuid = "job-completed"
    _create_standalone_presentation(slides_db, "presentation-1", job_uuid=job_uuid)
    _insert_receipt(
        slides_db,
        "receipt-completed",
        receipt_status="completed",
        job_id=7,
        job_uuid=job_uuid,
        presentation_id="presentation-1",
        expires_at="2026-08-18T12:00:00+00:00",
        include_input=False,
    )

    [row] = slides_db.list_generation_receipts_for_reconciliation(
        owner_user_id=_OWNER,
        after_receipt_id=None,
        limit=10,
    )

    assert row.presentation_exists is True
    assert row.presentation_content_kind == "standalone_html"
    assert row.presentation_generation_job_uuid == job_uuid
    assert row.input_exists is False


def test_repair_generation_receipt_job_uses_exact_uuid_cas_and_clears_missing_marker(slides_db):
    first_missing_at = "2026-07-18T12:05:00+00:00"
    _insert_receipt(
        slides_db,
        "receipt-repair",
        error_code="generation_receipt_unresolved_pending",
        updated_at=first_missing_at,
    )

    assert (
        slides_db.repair_generation_receipt_job(
            receipt_id="receipt-repair",
            owner_user_id="owner-2",
            expected_job_uuid=None,
            job_id=17,
            job_uuid="job-17",
            receipt_status="queued",
            updated_at="2026-07-18T12:06:00+00:00",
        )
        is False
    )
    assert slides_db.repair_generation_receipt_job(
        receipt_id="receipt-repair",
        owner_user_id=_OWNER,
        expected_job_uuid=None,
        job_id=17,
        job_uuid="job-17",
        receipt_status="queued",
        updated_at="2026-07-18T12:06:00+00:00",
    )
    repaired = slides_db.get_generation_receipt("receipt-repair", owner_user_id=_OWNER)
    assert (repaired.job_id, repaired.job_uuid, repaired.receipt_status) == (17, "job-17", "queued")
    assert (repaired.error_code, repaired.error_message) == (None, None)
    assert (
        slides_db.repair_generation_receipt_job(
            receipt_id="receipt-repair",
            owner_user_id=_OWNER,
            expected_job_uuid="different-job",
            job_id=18,
            job_uuid="different-job",
            receipt_status="running",
            updated_at="2026-07-18T12:07:00+00:00",
        )
        is False
    )
    assert slides_db.repair_generation_receipt_job(
        receipt_id="receipt-repair",
        owner_user_id=_OWNER,
        expected_job_uuid="job-17",
        job_id=17,
        job_uuid="job-17",
        receipt_status="running",
        updated_at="2026-07-18T12:07:00+00:00",
    )
    assert slides_db.get_generation_receipt("receipt-repair", owner_user_id=_OWNER).receipt_status == "running"


def test_repair_generation_receipt_job_rejects_status_or_uuid_replacement(slides_db):
    _insert_receipt(slides_db, "receipt-bound", job_id=3, job_uuid="job-3", receipt_status="queued")

    with pytest.raises(InputError, match="receipt status"):
        slides_db.repair_generation_receipt_job(
            receipt_id="receipt-bound",
            owner_user_id=_OWNER,
            expected_job_uuid="job-3",
            job_id=3,
            job_uuid="job-3",
            receipt_status="completed",
            updated_at=_CREATED_AT,
        )
    with pytest.raises(InputError, match="replace a Jobs UUID"):
        slides_db.repair_generation_receipt_job(
            receipt_id="receipt-bound",
            owner_user_id=_OWNER,
            expected_job_uuid="job-3",
            job_id=4,
            job_uuid="job-4",
            receipt_status="queued",
            updated_at=_CREATED_AT,
        )


def test_repair_generation_receipt_job_requires_a_positive_job_id(slides_db):
    _insert_receipt(slides_db, "receipt-zero-job-id")

    with pytest.raises(InputError, match="positive integer"):
        slides_db.repair_generation_receipt_job(
            receipt_id="receipt-zero-job-id",
            owner_user_id=_OWNER,
            expected_job_uuid=None,
            job_id=0,
            job_uuid="job-zero",
            receipt_status="queued",
            updated_at=_CREATED_AT,
        )


def test_first_confirmed_missing_marker_is_persistent(slides_db):
    _insert_receipt(slides_db, "receipt-missing", job_id=9, job_uuid="job-9", receipt_status="queued")
    first_observed_at = "2026-07-18T12:05:00+00:00"

    first = slides_db.mark_generation_receipt_job_missing(
        receipt_id="receipt-missing",
        owner_user_id=_OWNER,
        expected_job_uuid="job-9",
        observed_at=first_observed_at,
    )
    repeated = slides_db.mark_generation_receipt_job_missing(
        receipt_id="receipt-missing",
        owner_user_id=_OWNER,
        expected_job_uuid="job-9",
        observed_at="2026-07-18T12:14:00+00:00",
    )

    assert first == repeated == first_observed_at
    receipt = slides_db.get_generation_receipt("receipt-missing", owner_user_id=_OWNER)
    assert receipt.error_code == "generation_receipt_unresolved_pending"
    assert receipt.updated_at == first_observed_at
    assert (
        slides_db.mark_generation_receipt_job_missing(
            receipt_id="receipt-missing",
            owner_user_id=_OWNER,
            expected_job_uuid="other-job",
            observed_at="2026-07-18T12:20:00+00:00",
        )
        is None
    )


def test_authoritative_job_repair_clears_first_missing_marker(slides_db):
    _insert_receipt(slides_db, "receipt-found", job_id=4, job_uuid="job-4", receipt_status="queued")
    slides_db.mark_generation_receipt_job_missing(
        receipt_id="receipt-found",
        owner_user_id=_OWNER,
        expected_job_uuid="job-4",
        observed_at="2026-07-18T12:05:00+00:00",
    )

    assert slides_db.repair_generation_receipt_job(
        receipt_id="receipt-found",
        owner_user_id=_OWNER,
        expected_job_uuid="job-4",
        job_id=4,
        job_uuid="job-4",
        receipt_status="queued",
        updated_at="2026-07-18T12:06:00+00:00",
    )
    receipt = slides_db.get_generation_receipt("receipt-found", owner_user_id=_OWNER)
    assert (receipt.error_code, receipt.error_message) == (None, None)


def test_expired_input_terminalization_uses_the_absolute_deadline(slides_db):
    deadline = "2026-07-19T12:00:00+00:00"
    _insert_receipt(
        slides_db,
        "receipt-expired",
        job_id=23,
        job_uuid="job-23",
        receipt_status="running",
        input_expires_at=deadline,
    )

    assert slides_db.terminalize_expired_generation_receipt(
        receipt_id="receipt-expired",
        owner_user_id=_OWNER,
        expected_job_uuid="job-23",
        as_of="2026-07-21T08:00:00+00:00",
    )
    receipt = slides_db.get_generation_receipt("receipt-expired", owner_user_id=_OWNER)
    assert receipt.receipt_status == "failed"
    assert receipt.error_code == "generation_expired"
    assert receipt.updated_at == deadline
    assert receipt.expires_at == "2026-08-18T12:00:00+00:00"
    with pytest.raises(KeyError, match="slides_generation_input_not_found"):
        slides_db.get_generation_input("receipt-expired", owner_user_id=_OWNER)
    assert (
        slides_db.terminalize_expired_generation_receipt(
            receipt_id="receipt-expired",
            owner_user_id=_OWNER,
            expected_job_uuid="job-23",
            as_of="2026-07-22T08:00:00+00:00",
        )
        is False
    )


def test_expired_receipt_without_input_terminalizes_at_receipt_deadline(slides_db):
    _insert_receipt(
        slides_db,
        "receipt-expired-missing-input",
        job_id=26,
        job_uuid="job-26",
        receipt_status="queued",
    )
    with slides_db.transaction(immediate=True) as connection:
        connection.execute(
            "DELETE FROM slides_generation_inputs WHERE receipt_id = ?",
            ("receipt-expired-missing-input",),
        )

    assert slides_db.terminalize_expired_generation_receipt(
        receipt_id="receipt-expired-missing-input",
        owner_user_id=_OWNER,
        expected_job_uuid="job-26",
        as_of=_INPUT_EXPIRES_AT,
    )
    receipt = slides_db.get_generation_receipt(
        "receipt-expired-missing-input",
        owner_user_id=_OWNER,
    )
    assert (receipt.receipt_status, receipt.error_code) == (
        "failed",
        "generation_expired",
    )
    assert receipt.updated_at == _INPUT_EXPIRES_AT


@pytest.mark.parametrize(
    ("input_created_at", "input_expires_at"),
    [
        (_CREATED_AT, "not-a-timestamp"),
        ("2026-07-20T12:00:00+00:00", "2026-07-21T12:00:00+00:00"),
    ],
)
def test_expiry_terminalization_fails_closed_at_derived_deadline_for_corrupt_input_metadata(
    slides_db,
    input_created_at,
    input_expires_at,
):
    _insert_receipt(
        slides_db,
        "receipt-corrupt-deadline",
        job_id=25,
        job_uuid="job-25",
        receipt_status="running",
        input_created_at=input_created_at,
        input_expires_at=input_expires_at,
    )
    connection = slides_db.get_connection()
    input_reads: set[str] = set()
    forbidden = {"source_text", "system_prompt", "provenance_json"}

    def deny_source_reads(action, table, column, _database, _trigger):
        if action == sqlite3.SQLITE_READ and table == "slides_generation_inputs":
            input_reads.add(column)
            if column in forbidden:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(deny_source_reads)
    try:
        assert slides_db.terminalize_expired_generation_receipt(
            receipt_id="receipt-corrupt-deadline",
            owner_user_id=_OWNER,
            expected_job_uuid="job-25",
            as_of=_INPUT_EXPIRES_AT,
        )
    finally:
        connection.set_authorizer(None)

    assert {"created_at", "input_expires_at"} <= input_reads
    receipt = slides_db.get_generation_receipt("receipt-corrupt-deadline", owner_user_id=_OWNER)
    assert (receipt.receipt_status, receipt.error_code) == ("failed", "generation_expired")
    assert receipt.updated_at == _INPUT_EXPIRES_AT
    assert receipt.expires_at == "2026-08-18T12:00:00+00:00"
    with pytest.raises(KeyError, match="slides_generation_input_not_found"):
        slides_db.get_generation_input("receipt-corrupt-deadline", owner_user_id=_OWNER)


def test_expiry_terminalization_requires_owner_uuid_and_elapsed_deadline(slides_db):
    _insert_receipt(slides_db, "receipt-live", job_id=24, job_uuid="job-24", receipt_status="queued")

    for owner, job_uuid, as_of in (
        ("owner-2", "job-24", "2026-07-20T12:00:00+00:00"),
        (_OWNER, "other-job", "2026-07-20T12:00:00+00:00"),
        (_OWNER, "job-24", "2026-07-19T11:59:59+00:00"),
    ):
        assert (
            slides_db.terminalize_expired_generation_receipt(
                receipt_id="receipt-live",
                owner_user_id=owner,
                expected_job_uuid=job_uuid,
                as_of=as_of,
            )
            is False
        )
    assert slides_db.get_generation_receipt("receipt-live", owner_user_id=_OWNER).receipt_status == "queued"
    assert slides_db.get_generation_input("receipt-live", owner_user_id=_OWNER)


def test_terminal_input_cleanup_is_owner_scoped_and_idempotent(slides_db):
    _insert_receipt(slides_db, "receipt-terminal-input", receipt_status="failed")

    assert (
        slides_db.delete_terminal_generation_input(
            receipt_id="receipt-terminal-input",
            owner_user_id="owner-2",
        )
        is False
    )
    assert slides_db.get_generation_input("receipt-terminal-input", owner_user_id=_OWNER)
    assert slides_db.delete_terminal_generation_input(
        receipt_id="receipt-terminal-input",
        owner_user_id=_OWNER,
    )
    assert (
        slides_db.delete_terminal_generation_input(
            receipt_id="receipt-terminal-input",
            owner_user_id=_OWNER,
        )
        is False
    )


def test_expired_terminal_receipt_cleanup_preserves_presentations(slides_db):
    _create_standalone_presentation(slides_db, "presentation-retained", job_uuid="job-retained")
    _insert_receipt(
        slides_db,
        "receipt-expired-terminal",
        receipt_status="completed",
        job_id=31,
        job_uuid="job-retained",
        presentation_id="presentation-retained",
        expires_at="2026-07-18T11:59:59+00:00",
        include_input=False,
    )
    _insert_receipt(
        slides_db,
        "receipt-not-expired",
        receipt_status="failed",
        expires_at="2026-07-18T12:00:01+00:00",
        include_input=False,
    )
    _insert_receipt(
        slides_db,
        "receipt-nonterminal",
        receipt_status="queued",
        expires_at="2026-07-18T11:59:59+00:00",
        include_input=False,
    )

    deleted = slides_db.delete_expired_generation_receipts(
        owner_user_id=_OWNER,
        expires_before="2026-07-18T12:00:00+00:00",
        limit=10,
    )

    assert deleted == 1
    with pytest.raises(KeyError, match="slides_generation_receipt_not_found"):
        slides_db.get_generation_receipt("receipt-expired-terminal", owner_user_id=_OWNER)
    assert (
        slides_db.get_presentation_by_id("presentation-retained", include_deleted=True).generation_job_uuid
        == "job-retained"
    )
    assert slides_db.get_generation_receipt("receipt-not-expired", owner_user_id=_OWNER)
    assert slides_db.get_generation_receipt("receipt-nonterminal", owner_user_id=_OWNER)


def test_unexpired_digest_key_reference_count_covers_the_physical_database(slides_db):
    _insert_receipt(
        slides_db,
        "receipt-key-nonterminal",
        digest_key_id="retiring-key",
        expires_at="2026-07-17T00:00:00+00:00",
        include_input=False,
    )
    _insert_receipt(
        slides_db,
        "receipt-key-future",
        digest_key_id="retiring-key",
        receipt_status="failed",
        expires_at="2026-07-19T00:00:00+00:00",
        include_input=False,
    )
    _insert_receipt(
        slides_db,
        "receipt-key-expired",
        digest_key_id="retiring-key",
        receipt_status="cancelled",
        expires_at="2026-07-17T00:00:00+00:00",
        include_input=False,
    )
    _insert_receipt(
        slides_db,
        "receipt-other-key",
        digest_key_id="current-key",
        include_input=False,
    )
    _insert_receipt(
        slides_db,
        "receipt-other-owner",
        owner_user_id="owner-2",
        digest_key_id="retiring-key",
        include_input=False,
    )

    assert (
        slides_db.count_unexpired_generation_receipts_for_digest_key(
            owner_user_id=_OWNER,
            digest_key_id="retiring-key",
            as_of="2026-07-18T12:00:00+00:00",
        )
        == 3
    )


def test_reconciliation_owner_id_is_required_for_physical_database_checks(slides_db):
    with pytest.raises(InputError, match="owner_user_id"):
        slides_db.list_generation_receipts_for_reconciliation(
            owner_user_id="",
            after_receipt_id=None,
            limit=10,
        )
    with pytest.raises(InputError, match="owner_user_id"):
        slides_db.count_unexpired_generation_receipts_for_digest_key(
            owner_user_id="",
            digest_key_id="retiring-key",
            as_of="2026-07-18T12:00:00+00:00",
        )


def test_reconciliation_timestamp_inputs_must_be_canonical_utc(slides_db):
    _insert_receipt(slides_db, "receipt-time", job_uuid="job-time", receipt_status="queued")

    with pytest.raises(InputError, match="canonical UTC"):
        slides_db.terminalize_expired_generation_receipt(
            receipt_id="receipt-time",
            owner_user_id=_OWNER,
            expected_job_uuid="job-time",
            as_of="2026-07-20T12:00:00",
        )
    with pytest.raises(InputError, match="canonical UTC"):
        slides_db.delete_expired_generation_receipts(
            owner_user_id=_OWNER,
            expires_before=datetime(2026, 7, 20, tzinfo=timezone(timedelta(hours=1))).isoformat(),
            limit=10,
        )
