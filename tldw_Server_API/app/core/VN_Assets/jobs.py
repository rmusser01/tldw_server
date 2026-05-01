"""Jobs payload helpers for VN asset generation."""

from __future__ import annotations

import os
from typing import Any

VN_ASSETS_DOMAIN = "vn_assets"
VN_ASSET_ENQUEUE_BATCH_JOB_TYPE = "vn_asset_enqueue_batch"
VN_ASSET_GENERATE_VARIANT_JOB_TYPE = "vn_asset_generate_variant"


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


def vn_asset_batch_group(*, user_id: int, pack_id: int, batch_id: int) -> str:
    return f"vn_assets:user:{int(user_id)}:pack:{int(pack_id)}:batch:{int(batch_id)}"


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
