"""Jobs helpers for visual identity expression pack imports."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

VISUAL_IDENTITIES_DOMAIN = "visual_identities"
VISUAL_IDENTITY_IMPORT_ZIP_JOB_TYPE = "visual_identity_import_zip"


def visual_identity_jobs_queue() -> str:
    """Return the queue used for visual identity import work."""
    return (os.getenv("VISUAL_IDENTITY_JOBS_QUEUE") or "default").strip() or "default"


def build_visual_identity_import_zip_payload(
    *,
    owner_user_id: int,
    draft_id: int,
    upload_path: str,
    source_filename: str,
) -> dict[str, Any]:
    """Build the JSON payload for a visual identity ZIP import job."""
    payload = {
        "owner_user_id": int(owner_user_id),
        "draft_id": int(draft_id),
        "upload_path": str(upload_path),
        "source_filename": str(source_filename),
    }
    payload["payload_hash"] = visual_identity_import_zip_payload_hash(payload)
    return payload


def visual_identity_import_zip_payload_hash(payload: dict[str, Any]) -> str:
    """Return a stable digest for idempotency-sensitive import payload fields."""
    normalized = {
        "draft_id": int(payload["draft_id"]),
        "owner_user_id": int(payload["owner_user_id"]),
        "source_filename": str(payload["source_filename"]),
        "upload_path": str(payload["upload_path"]),
    }
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def visual_identity_import_zip_group(*, owner_user_id: int, draft_id: int) -> str:
    """Return the batch group for one user's draft import."""
    return f"{VISUAL_IDENTITIES_DOMAIN}:user:{int(owner_user_id)}:draft:{int(draft_id)}:import"


def visual_identity_import_zip_idempotency_key(
    *,
    owner_user_id: int,
    draft_id: int,
    upload_path: str,
    source_filename: str,
) -> str:
    """Build a deterministic idempotency key for a ZIP import request."""
    payload = build_visual_identity_import_zip_payload(
        owner_user_id=owner_user_id,
        draft_id=draft_id,
        upload_path=upload_path,
        source_filename=source_filename,
    )
    return (
        f"{visual_identity_import_zip_group(owner_user_id=owner_user_id, draft_id=draft_id)}:"
        f"{payload['payload_hash'][:16]}"
    )


def create_visual_identity_import_zip_job(
    jobs_manager: Any,
    *,
    owner_user_id: int,
    draft_id: int,
    upload_path: str,
    source_filename: str,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Create a Jobs row for importing a ZIP into a visual identity draft."""
    payload = build_visual_identity_import_zip_payload(
        owner_user_id=owner_user_id,
        draft_id=draft_id,
        upload_path=upload_path,
        source_filename=source_filename,
    )
    return jobs_manager.create_job(
        domain=VISUAL_IDENTITIES_DOMAIN,
        queue=visual_identity_jobs_queue(),
        job_type=VISUAL_IDENTITY_IMPORT_ZIP_JOB_TYPE,
        payload=payload,
        owner_user_id=str(owner_user_id),
        batch_group=visual_identity_import_zip_group(
            owner_user_id=owner_user_id,
            draft_id=draft_id,
        ),
        idempotency_key=(
            str(idempotency_key)
            if idempotency_key
            else visual_identity_import_zip_idempotency_key(
                owner_user_id=owner_user_id,
                draft_id=draft_id,
                upload_path=upload_path,
                source_filename=source_filename,
            )
        ),
        max_retries=2,
    )


__all__ = [
    "VISUAL_IDENTITIES_DOMAIN",
    "VISUAL_IDENTITY_IMPORT_ZIP_JOB_TYPE",
    "build_visual_identity_import_zip_payload",
    "create_visual_identity_import_zip_job",
    "visual_identity_import_zip_group",
    "visual_identity_import_zip_idempotency_key",
    "visual_identity_import_zip_payload_hash",
    "visual_identity_jobs_queue",
]
