"""Lease expiry arithmetic binds the interval as a parameter and still lands on the right time.

The lease length used to be f-string interpolated into the SQL; it is now bound. These
pin the resulting ``leased_until`` on both backends, including the three statements
that carry it (acquire, update-to-processing, renew) and their placeholder ordering.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

_TOLERANCE_SECONDS = 5


def _utc(value: Any) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _lease_length(job: dict[str, Any]) -> float:
    return (_utc(job["leased_until"]) - _utc(job["started_at"])).total_seconds()


def test_acquire_sets_lease_to_configured_seconds(prompt_studio_dual_backend_db, monkeypatch):
    _label, db = prompt_studio_dual_backend_db
    monkeypatch.setenv("TLDW_PS_JOB_LEASE_SECONDS", "120")
    db.create_job("evaluation", 1, {})

    job = db.acquire_next_job(worker_id="w1")

    assert job is not None
    assert job["lease_owner"] == "w1"
    assert abs(_lease_length(job) - 120) <= _TOLERANCE_SECONDS


def test_update_to_processing_sets_lease(prompt_studio_dual_backend_db, monkeypatch):
    _label, db = prompt_studio_dual_backend_db
    monkeypatch.setenv("TLDW_PS_JOB_LEASE_SECONDS", "90")
    created = db.create_job("evaluation", 1, {})

    job = db.update_job_status(created["id"], "processing", error_message="kept")

    assert job["error_message"] == "kept"
    assert abs(_lease_length(job) - 90) <= _TOLERANCE_SECONDS


def test_renew_extends_live_lease_by_bound_seconds(prompt_studio_dual_backend_db, monkeypatch):
    _label, db = prompt_studio_dual_backend_db
    monkeypatch.setenv("TLDW_PS_JOB_LEASE_SECONDS", "60")
    db.create_job("evaluation", 1, {})
    job = db.acquire_next_job(worker_id="w1")
    before = _utc(job["leased_until"])

    # renew_job_lease swallows DB errors as False, so a placeholder/param mismatch
    # would surface here as a False return rather than an exception.
    assert db.renew_job_lease(job["id"], seconds=600, worker_id="w1") is True

    after = _utc(db.get_job(job["id"])["leased_until"])
    assert abs((after - before).total_seconds() - 600) <= _TOLERANCE_SECONDS
