"""Jobs payload helpers for VN asset generation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import LegacyActivityReader
from tldw_Server_API.app.core.exceptions import VNAssetGenerationError

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


def recover_enqueue_batch_job(
    jobs_manager: Any,
    *,
    job_batch_id: str | None,
    pack_id: int,
    batch_id: int,
    user_id: int,
    fanout_complete: bool,
) -> dict[str, Any]:
    """Resolve an exact owned parent; admit missing or failed incomplete fanout.

    Only explicit unfinished receipt recovery calls this. Deterministic create
    can replay a failed row, so every returned ID is read through owner-scoped
    Jobs lookup before using supported retry admission. Admin-cancelled,
    quarantined and incomplete completed parents stay pending. Jobs outages
    propagate; policy or identity rejections use the existing retryable code.
    """
    payload = build_enqueue_batch_payload(pack_id=pack_id, batch_id=batch_id, user_id=user_id)
    queue = vn_asset_jobs_queue()
    key = enqueue_batch_idempotency_key(user_id=user_id, pack_id=pack_id, batch_id=batch_id)

    def pending() -> VNAssetGenerationError:
        """Keep an unhealthy receipt unfinished without changing public codes."""
        return VNAssetGenerationError(
            "idempotency_key_in_progress", retryable=True, pack_id=pack_id, batch_id=batch_id,
            operation="recover_generation_parent",
        )

    def read_parent(identity: Any) -> dict[str, Any] | None:
        """Require a positive numeric ID and validate the authoritative owned row."""
        if isinstance(identity, bool) or not str(identity).isdigit() or int(identity) <= 0:
            raise pending()
        row = jobs_manager.get_job(int(identity), owner_user_id=str(user_id))
        if row is not None and (
            str(row.get("id")) != str(identity)
            or row.get("owner_user_id") != str(user_id)
            or row.get("domain") != VN_ASSETS_DOMAIN
            or row.get("queue") != queue
            or row.get("job_type") != VN_ASSET_ENQUEUE_BATCH_JOB_TYPE
            or row.get("idempotency_key") != key
            or row.get("payload") != payload
            or any(type(value) is not int for value in row.get("payload", {}).values())
            or row.get("cancel_requested_at") is not None
        ):
            raise pending()
        if row is not None and row.get("status") == "processing" and not jobs_manager.has_live_processing_lease(
            int(row["id"]), owner_user_id=str(user_id),
        ):
            raise pending()
        return row

    parent = read_parent(job_batch_id) if job_batch_id else None
    if parent is None:
        if fanout_complete:
            raise pending()
        created = create_enqueue_batch_job(jobs_manager, pack_id=pack_id, batch_id=batch_id, user_id=user_id)
        parent = read_parent(created.get("id"))
        if parent is None:
            raise pending()
    status = parent.get("status")
    if status in {"queued", "processing"}:
        return parent
    if fanout_complete and status in {"completed", "failed"}:
        return parent
    if status != "failed":
        raise pending()
    try:
        retried = jobs_manager.retry_failed_job_admission(
            job_id=int(parent["id"]), owner_user_id=str(user_id), expected_uuid=parent["uuid"],
            domain=VN_ASSETS_DOMAIN, queue=queue, job_type=VN_ASSET_ENQUEUE_BATCH_JOB_TYPE,
            expected_payload=payload, idempotency_key=key,
        )
    except ValueError as exc:
        raise pending() from exc
    authoritative = read_parent(retried.get("id"))
    if authoritative is None or authoritative.get("status") not in {"queued", "processing"}:
        raise pending()
    return authoritative


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


def legacy_delivery_fingerprint(job: Mapping[str, Any] | None) -> str | None:
    """Identify an exact delivery without persisting its lease token or authority."""
    if job is None:
        return None
    job_id, job_uuid, lease_id = job.get("id"), job.get("uuid"), job.get("lease_id")
    if type(job_id) is not int or job_id <= 0 or not all(
        isinstance(value, str) and value for value in (job_uuid, lease_id)
    ):
        return None
    identity = json.dumps(["vn-assets-legacy-delivery-v1", job_id, job_uuid, lease_id], separators=(",", ":"))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_legacy_activity_reader(jobs_manager: Any) -> LegacyActivityReader:
    """Build a read-only display callback; Jobs retains all lease authority.

    Scan canonical legacy batch children with stable Jobs pagination, then
    re-read each relevant identity owner-scoped. Only unexpired processing
    leases without cancellation requests generate; queued children queue only
    in nonterminal VN batches. No legacy admission/outcome contract changes.
    Read failures propagate to the caller's write-locked VN transaction.
    The optional finishing ID/lease excludes only one caller's SDK handoff,
    not its replacement attempt or any live sibling. Published provenance
    settles only the exact current delivery, never unknown historical work.
    """
    def read(
        pack_id: int, slot_id: int, user_id: int, batches: Mapping[int, str],
        settled: set[tuple[int, int, str]], finishing_delivery: tuple[int, str] | None,
    ) -> tuple[bool, bool]:
        """Return exact active/queued activity, never inferred from counters."""
        for status in ("processing", "queued"):
            for batch_id, batch_status in batches.items():
                if batch_status == "cancelled" or (status == "queued" and batch_status in {"failed", "completed"}):
                    continue
                group = vn_asset_batch_group(user_id=user_id, pack_id=pack_id, batch_id=batch_id)
                cursor: dict[str, Any] = {}
                while True:
                    rows = jobs_manager.list_jobs(
                        domain=VN_ASSETS_DOMAIN, queue=vn_asset_generation_jobs_queue(),
                        job_type=VN_ASSET_GENERATE_VARIANT_JOB_TYPE, owner_user_id=str(user_id),
                        batch_group=group, status=status, sort_by="created_at", sort_order="desc",
                        limit=100, **cursor,
                    )
                    for row in rows:
                        payload = row.get("payload")
                        if not isinstance(payload, dict) or type(payload.get("variant_index")) is not int:
                            continue
                        index = payload["variant_index"]
                        if index < 0 or payload.get("slot_id") != slot_id:
                            continue
                        current = jobs_manager.get_job(int(row["id"]), owner_user_id=str(user_id))
                        expected = build_generate_variant_payload(
                            pack_id=pack_id, slot_id=slot_id, batch_id=batch_id, variant_index=index, user_id=user_id,
                        )
                        if current is None or (
                            current.get("id") != row["id"] or current.get("uuid") != row.get("uuid")
                            or current.get("owner_user_id") != str(user_id)
                            or current.get("domain") != VN_ASSETS_DOMAIN
                            or current.get("queue") != vn_asset_generation_jobs_queue()
                            or current.get("job_type") != VN_ASSET_GENERATE_VARIANT_JOB_TYPE
                            or current.get("batch_group") != group or current.get("payload") != expected
                            or any(type(value) is not int for value in current.get("payload", {}).values())
                            or current.get("idempotency_key") != generate_variant_idempotency_key(**expected)
                            or current.get("status") != status or current.get("cancel_requested_at")
                        ):
                            continue
                        fingerprint = legacy_delivery_fingerprint(current)
                        if fingerprint is not None and (batch_id, index, fingerprint) in settled:
                            continue
                        if status == "queued":
                            return False, True
                        if finishing_delivery == (int(current["id"]), current.get("lease_id")):
                            continue
                        if jobs_manager.has_live_processing_lease(int(current["id"]), owner_user_id=str(user_id)):
                            return True, False
                    if len(rows) < 100:
                        break
                    last = rows[-1]
                    next_cursor = {
                        "created_before": datetime.fromisoformat(str(last["created_at"]).replace("Z", "+00:00")),
                        "before_id": int(last["id"]),
                    }
                    if next_cursor == cursor:
                        raise RuntimeError("vn_asset_legacy_jobs_cursor_stalled")
                    cursor = next_cursor
        return False, False

    return read
