from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.Visual_Identities.jobs import (
    VISUAL_IDENTITIES_DOMAIN,
    VISUAL_IDENTITY_IMPORT_ZIP_JOB_TYPE,
    create_visual_identity_import_zip_job,
)


class RecordingJobsManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"id": "job-1", **kwargs}


def test_create_visual_identity_import_zip_job_uses_expected_jobs_contract(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VISUAL_IDENTITY_JOBS_QUEUE", "identity-imports")
    manager = RecordingJobsManager()

    job = create_visual_identity_import_zip_job(
        manager,
        owner_user_id=42,
        draft_id=7,
        upload_path="/tmp/imports/pack.zip",
        source_filename="pack.zip",
    )

    assert job["domain"] == VISUAL_IDENTITIES_DOMAIN == "visual_identities"
    assert job["job_type"] == VISUAL_IDENTITY_IMPORT_ZIP_JOB_TYPE == "visual_identity_import_zip"
    assert job["queue"] == "identity-imports"
    assert job["owner_user_id"] == "42"
    assert job["batch_group"] == "visual_identities:user:42:draft:7:import"
    assert job["payload"]["owner_user_id"] == 42
    assert job["payload"]["draft_id"] == 7
    assert job["payload"]["upload_path"] == "/tmp/imports/pack.zip"
    assert job["payload"]["source_filename"] == "pack.zip"
    assert len(job["payload"]["payload_hash"]) == 64
    assert job["idempotency_key"].startswith("visual_identities:user:42:draft:7:import:")


def test_visual_identity_import_zip_idempotency_is_stable_and_payload_sensitive() -> None:
    first = create_visual_identity_import_zip_job(
        RecordingJobsManager(),
        owner_user_id=42,
        draft_id=7,
        upload_path="/tmp/imports/pack.zip",
        source_filename="pack.zip",
    )
    second = create_visual_identity_import_zip_job(
        RecordingJobsManager(),
        owner_user_id=42,
        draft_id=7,
        upload_path="/tmp/imports/pack.zip",
        source_filename="pack.zip",
    )
    changed_source = create_visual_identity_import_zip_job(
        RecordingJobsManager(),
        owner_user_id=42,
        draft_id=7,
        upload_path="/tmp/imports/pack.zip",
        source_filename="other.zip",
    )
    changed_upload = create_visual_identity_import_zip_job(
        RecordingJobsManager(),
        owner_user_id=42,
        draft_id=7,
        upload_path="/tmp/imports/other.zip",
        source_filename="pack.zip",
    )
    explicit = create_visual_identity_import_zip_job(
        RecordingJobsManager(),
        owner_user_id=42,
        draft_id=7,
        upload_path="/tmp/imports/pack.zip",
        source_filename="pack.zip",
        idempotency_key="client-key",
    )

    assert first["idempotency_key"] == second["idempotency_key"]
    assert first["payload"]["payload_hash"] == second["payload"]["payload_hash"]
    assert changed_source["idempotency_key"] != first["idempotency_key"]
    assert changed_upload["idempotency_key"] != first["idempotency_key"]
    assert explicit["idempotency_key"] == "client-key"
