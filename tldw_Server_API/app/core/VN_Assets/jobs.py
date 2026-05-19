"""Jobs payload helpers for VN asset generation."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

VN_ASSETS_DOMAIN = "vn_assets"
VN_ASSET_ENQUEUE_BATCH_JOB_TYPE = "vn_asset_enqueue_batch"
VN_ASSET_GENERATE_VARIANT_JOB_TYPE = "vn_asset_generate_variant"
VN_PACK_EXPORT_JOB_TYPE = "vn_pack_export"
VN_PACK_IMPORT_PREVIEW_JOB_TYPE = "vn_pack_import_preview"
VN_PACK_IMPORT_COMMIT_JOB_TYPE = "vn_pack_import_commit"


def vn_asset_jobs_queue() -> str:
    queue = (os.getenv("VN_ASSET_JOBS_QUEUE") or "default").strip()
    return queue or "default"


def vn_asset_generation_jobs_queue() -> str:
    queue = (os.getenv("VN_ASSET_GENERATION_JOBS_QUEUE") or "generation").strip()
    return queue or "generation"


def build_enqueue_batch_payload(*, pack_id: int, batch_id: int, user_id: int) -> dict[str, int]:
    return {
        "pack_id": int(pack_id),
        "batch_id": int(batch_id),
        "user_id": int(user_id),
    }


def build_generate_variant_payload(
    *,
    pack_id: int,
    slot_id: int,
    variant_index: int,
    batch_id: int,
    user_id: int,
) -> dict[str, int]:
    return {
        "pack_id": int(pack_id),
        "slot_id": int(slot_id),
        "variant_index": int(variant_index),
        "batch_id": int(batch_id),
        "user_id": int(user_id),
    }


def build_pack_export_payload(
    *,
    pack_id: int,
    portability_job_id: int,
    request_id: str,
    user_id: int,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "pack_id": int(pack_id),
        "portability_job_id": int(portability_job_id),
        "request_id": str(request_id),
        "user_id": int(user_id),
        "options": dict(options or {}),
    }


def build_pack_import_preview_payload(
    *,
    preview_id: int,
    archive_path: str,
    request_id: str,
    user_id: int,
) -> dict[str, Any]:
    return {
        "preview_id": int(preview_id),
        "archive_path": str(archive_path),
        "request_id": str(request_id),
        "user_id": int(user_id),
    }


def build_pack_import_commit_payload(
    *,
    import_id: int,
    preview_id: int,
    request_id: str,
    user_id: int,
    trust_mode: str,
    target_mode: str,
    character_action: str,
    target_character_id: int | None = None,
    target_pack_id: int | None = None,
    conflict_decisions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "import_id": int(import_id),
        "preview_id": int(preview_id),
        "request_id": str(request_id),
        "user_id": int(user_id),
        "trust_mode": str(trust_mode),
        "target_mode": str(target_mode),
        "character_action": str(character_action),
        "target_character_id": None if target_character_id is None else int(target_character_id),
        "target_pack_id": None if target_pack_id is None else int(target_pack_id),
        "conflict_decisions": dict(conflict_decisions or {}),
    }


def vn_asset_batch_group(*, user_id: int, pack_id: int, batch_id: int) -> str:
    return f"vn_assets:user:{int(user_id)}:pack:{int(pack_id)}:batch:{int(batch_id)}"


def vn_pack_export_group(*, user_id: int, pack_id: int, request_id: str) -> str:
    return f"vn_assets:user:{int(user_id)}:pack:{int(pack_id)}:portability:export:{str(request_id)}"


def vn_pack_import_preview_group(*, user_id: int, preview_id: int, request_id: str) -> str:
    return f"vn_assets:user:{int(user_id)}:portability:import-preview:{int(preview_id)}:{str(request_id)}"


def vn_pack_import_commit_group(*, user_id: int, preview_id: int, import_id: int, request_id: str) -> str:
    return (
        f"vn_assets:user:{int(user_id)}:portability:import-commit:"
        f"{int(preview_id)}:{int(import_id)}:{str(request_id)}"
    )


def enqueue_batch_idempotency_key(*, user_id: int, pack_id: int, batch_id: int) -> str:
    return f"{vn_asset_batch_group(user_id=user_id, pack_id=pack_id, batch_id=batch_id)}:enqueue"


def generate_variant_idempotency_key(
    *,
    user_id: int,
    pack_id: int,
    batch_id: int,
    slot_id: int,
    variant_index: int,
) -> str:
    return (
        f"{vn_asset_batch_group(user_id=user_id, pack_id=pack_id, batch_id=batch_id)}"
        f":slot:{int(slot_id)}:variant:{int(variant_index)}"
    )


def pack_export_idempotency_key(
    *,
    user_id: int,
    pack_id: int,
    request_id: str,
    options: dict[str, Any] | None = None,
) -> str:
    options_digest = hashlib.sha256(
        json.dumps(dict(options or {}), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return (
        f"{vn_pack_export_group(user_id=user_id, pack_id=pack_id, request_id=request_id)}"
        f":{options_digest}"
    )


def pack_import_preview_idempotency_key(
    *,
    user_id: int,
    preview_id: int,
    request_id: str,
    archive_path: str,
) -> str:
    archive_digest = hashlib.sha256(str(archive_path).encode("utf-8")).hexdigest()[:16]
    return (
        f"{vn_pack_import_preview_group(user_id=user_id, preview_id=preview_id, request_id=request_id)}"
        f":{archive_digest}"
    )


def pack_import_commit_idempotency_key(
    *,
    user_id: int,
    preview_id: int,
    import_id: int,
    request_id: str,
) -> str:
    return vn_pack_import_commit_group(
        user_id=user_id,
        preview_id=preview_id,
        import_id=import_id,
        request_id=request_id,
    )


def create_enqueue_batch_job(
    jobs_manager: Any,
    *,
    pack_id: int,
    batch_id: int,
    user_id: int,
) -> dict[str, Any]:
    return jobs_manager.create_job(
        domain=VN_ASSETS_DOMAIN,
        queue=vn_asset_jobs_queue(),
        job_type=VN_ASSET_ENQUEUE_BATCH_JOB_TYPE,
        payload=build_enqueue_batch_payload(pack_id=pack_id, batch_id=batch_id, user_id=user_id),
        owner_user_id=str(user_id),
        batch_group=vn_asset_batch_group(user_id=user_id, pack_id=pack_id, batch_id=batch_id),
        idempotency_key=enqueue_batch_idempotency_key(
            user_id=user_id,
            pack_id=pack_id,
            batch_id=batch_id,
        ),
        max_retries=3,
    )


def create_pack_export_job(
    jobs_manager: Any,
    *,
    pack_id: int,
    portability_job_id: int,
    request_id: str,
    user_id: int,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return jobs_manager.create_job(
        domain=VN_ASSETS_DOMAIN,
        queue=vn_asset_jobs_queue(),
        job_type=VN_PACK_EXPORT_JOB_TYPE,
        payload=build_pack_export_payload(
            pack_id=pack_id,
            portability_job_id=portability_job_id,
            request_id=request_id,
            user_id=user_id,
            options=options,
        ),
        owner_user_id=str(user_id),
        batch_group=vn_pack_export_group(
            user_id=user_id,
            pack_id=pack_id,
            request_id=request_id,
        ),
        idempotency_key=pack_export_idempotency_key(
            user_id=user_id,
            pack_id=pack_id,
            request_id=request_id,
            options=options,
        ),
        max_retries=2,
    )


def create_pack_import_preview_job(
    jobs_manager: Any,
    *,
    preview_id: int,
    archive_path: str,
    request_id: str,
    user_id: int,
) -> dict[str, Any]:
    return jobs_manager.create_job(
        domain=VN_ASSETS_DOMAIN,
        queue=vn_asset_jobs_queue(),
        job_type=VN_PACK_IMPORT_PREVIEW_JOB_TYPE,
        payload=build_pack_import_preview_payload(
            preview_id=preview_id,
            archive_path=archive_path,
            request_id=request_id,
            user_id=user_id,
        ),
        owner_user_id=str(user_id),
        batch_group=vn_pack_import_preview_group(
            user_id=user_id,
            preview_id=preview_id,
            request_id=request_id,
        ),
        idempotency_key=pack_import_preview_idempotency_key(
            user_id=user_id,
            preview_id=preview_id,
            request_id=request_id,
            archive_path=archive_path,
        ),
        max_retries=2,
    )


def create_pack_import_commit_job(
    jobs_manager: Any,
    *,
    import_id: int,
    preview_id: int,
    request_id: str,
    user_id: int,
    trust_mode: str,
    target_mode: str,
    character_action: str,
    target_character_id: int | None = None,
    target_pack_id: int | None = None,
    conflict_decisions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    batch_group = vn_pack_import_commit_group(
        user_id=user_id,
        preview_id=preview_id,
        import_id=import_id,
        request_id=request_id,
    )
    return jobs_manager.create_job(
        domain=VN_ASSETS_DOMAIN,
        queue=vn_asset_jobs_queue(),
        job_type=VN_PACK_IMPORT_COMMIT_JOB_TYPE,
        payload=build_pack_import_commit_payload(
            import_id=import_id,
            preview_id=preview_id,
            request_id=request_id,
            user_id=user_id,
            trust_mode=trust_mode,
            target_mode=target_mode,
            character_action=character_action,
            target_character_id=target_character_id,
            target_pack_id=target_pack_id,
            conflict_decisions=conflict_decisions,
        ),
        owner_user_id=str(user_id),
        batch_group=batch_group,
        idempotency_key=batch_group,
        max_retries=1,
    )


def create_generate_variant_job(
    jobs_manager: Any,
    *,
    pack_id: int,
    slot_id: int,
    variant_index: int,
    batch_id: int,
    user_id: int,
) -> dict[str, Any]:
    return jobs_manager.create_job(
        domain=VN_ASSETS_DOMAIN,
        queue=vn_asset_generation_jobs_queue(),
        job_type=VN_ASSET_GENERATE_VARIANT_JOB_TYPE,
        payload=build_generate_variant_payload(
            pack_id=pack_id,
            slot_id=slot_id,
            variant_index=variant_index,
            batch_id=batch_id,
            user_id=user_id,
        ),
        owner_user_id=str(user_id),
        batch_group=vn_asset_batch_group(user_id=user_id, pack_id=pack_id, batch_id=batch_id),
        idempotency_key=generate_variant_idempotency_key(
            user_id=user_id,
            pack_id=pack_id,
            batch_id=batch_id,
            slot_id=slot_id,
            variant_index=variant_index,
        ),
        max_retries=1,
    )
