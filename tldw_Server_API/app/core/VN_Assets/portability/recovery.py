"""Discard work attributable to an owner-scoped VN pack import checkpoint."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from inspect import isawaitable
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Storage.generated_file_helpers import (
    remove_stale_vn_import_parts,
    vn_import_asset_storage_ref,
)
from tldw_Server_API.app.core.VN_Assets.storage import (
    generated_file_matches_vn_asset,
    resolve_vn_asset_storage_path,
    vn_asset_source_ref,
)


class ImportCleanupConflict(ValueError):
    """Some rows cannot safely be removed from this import."""


def legacy_import_may_have_written(records: Mapping[str, Any]) -> bool:
    """Check for writes that a journal without a checkpoint cannot attribute."""
    return records.get("writes_started") is True or any(
        records.get(field) for field in ("pack_id", "slot_ids", "item_ids", "generated_file_ids")
    )


async def discard_import_checkpoint(
    repo: VNAssetPacksRepository,
    *,
    import_id: int,
    owner_user_id: int,
    abandoned: bool,
    storage: Any = None,
    unregister: Callable[..., Any] | None = None,
    require_delivery: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Remove an interrupted import's own files and metadata, including unattached bytes."""
    with repo.db.transaction():
        if require_delivery is not None:
            require_delivery()
        journal = repo.get_import_journal(import_id, owner_user_id=owner_user_id)
        if journal is None or journal["completed_at"] or journal["status"] == "completed":
            raise ImportCleanupConflict("import_cleanup_requires_incomplete_journal")
        previous = json.loads(journal["cleanup_status_json"])
        if previous.get("status") == "discarded":
            return previous
        if previous.get("status") == "discarding" and previous.get("abandoned") and not abandoned:
            raise ImportCleanupConflict("import_cleanup_already_requested")
        records = json.loads(journal["created_records_json"])
        if records.get("checkpoint_version") != 1:
            result = {"status": "manual_required" if legacy_import_may_have_written(records) else "nothing_to_clean"}
        elif previous.get("status") == "completed":
            result = {**previous, "status": "discarded"} if abandoned else previous
        else:
            result = {"status": "discarding", "abandoned": abandoned}
            _validate_rows(repo, journal, records)
        repo.update_import_journal(import_id, {"cleanup_status": result}, owner_user_id=owner_user_id)
    if result["status"] != "discarding":
        return result

    unregistered = []
    if storage is not None:
        for raw_item_id, image_format in records["asset_formats"].items():
            item_id = int(raw_item_id)
            file_id = await _discard_import_asset(
                repo,
                storage=storage,
                storage_ref=vn_import_asset_storage_ref(import_id, item_id, image_format),
                item_id=item_id,
                owner_user_id=owner_user_id,
                require_delivery=require_delivery,
            )
            if file_id is not None:
                unregistered.append(file_id)
    elif records["generated_file_ids"]:
        if unregister is None:
            raise ImportCleanupConflict("import_cleanup_storage_unavailable")
        for file_id in records["generated_file_ids"]:
            if require_delivery is not None:
                require_delivery()
            result_value = unregister(file_id, hard_delete=True)
            removed = await result_value if isawaitable(result_value) else result_value
            if removed:
                unregistered.append(file_id)

    with repo.db.transaction():
        if require_delivery is not None:
            require_delivery()
        current = repo.get_import_journal(import_id, owner_user_id=owner_user_id)
        active = json.loads(current["cleanup_status_json"]) if current else {}
        if active.get("status") in {"completed", "discarded"}:
            return active
        if current is None or active.get("status") != "discarding" or current["completed_at"]:
            raise ImportCleanupConflict("import_cleanup_state_changed")
        _validate_rows(repo, current, records)
        for item_id in reversed(records["item_ids"]):
            repo.delete_item(item_id)
        for slot_id in sorted(records["slot_ids"], reverse=True):
            repo.delete_slot(slot_id)
        new_pack_id = records["pack_id"] if journal["target_mode"] == "create_new" else None
        if new_pack_id is not None:
            repo.soft_delete_pack(new_pack_id)
        result = {
            "status": "discarded" if abandoned else "completed",
            "generated_file_ids": records["generated_file_ids"],
            "unregistered_generated_file_ids": unregistered,
            "soft_deleted_pack_id": new_pack_id,
        }
        repo.update_import_journal(import_id, {"cleanup_status": result}, owner_user_id=owner_user_id)
    return result


def _validate_rows(repo: VNAssetPacksRepository, journal: Mapping[str, Any], records: Mapping[str, Any]) -> None:
    pack_id = int(records["pack_id"])
    pack = repo.get_pack(pack_id)
    if pack is None or pack["owner_user_id"] != journal["owner_user_id"]:
        raise ImportCleanupConflict("import_cleanup_pack_mismatch")
    owned_slots = set(records["slot_ids"])
    owned_items = set(records["item_ids"])
    if any(int(item_id) not in owned_items for item_id in records["asset_formats"]):
        raise ImportCleanupConflict("import_cleanup_item_mismatch")
    for slot in repo.list_slots(pack_id, include_import_staging=True):
        if slot["id"] not in owned_slots and (
            journal["target_mode"] == "create_new" or slot["depends_on_slot_id"] in owned_slots
        ):
            raise ImportCleanupConflict("import_cleanup_pack_has_other_slots")
    for item in repo.list_items(pack_id, include_import_staging=True):
        if item["id"] in owned_items:
            context = json.loads(item.get("source_context_snapshot_json") or "{}")
            if context.get("vnpack_import", {}).get("import_id") != journal["id"]:
                raise ImportCleanupConflict("import_cleanup_item_mismatch")
        elif (
            journal["target_mode"] == "create_new"
            or item["slot_id"] in owned_slots
            or item["parent_item_id"] in owned_items
        ):
            raise ImportCleanupConflict("import_cleanup_pack_has_other_items")
    for slot_id in owned_slots:
        slot = repo.get_slot(slot_id)
        if slot is not None and slot["pack_id"] != pack_id:
            raise ImportCleanupConflict("import_cleanup_slot_mismatch")
    for item_id in owned_items:
        item = repo.get_item(item_id)
        if item is not None and item["pack_id"] != pack_id:
            raise ImportCleanupConflict("import_cleanup_item_mismatch")


async def _discard_import_asset(
    repo: VNAssetPacksRepository,
    *,
    storage: Any,
    storage_ref: str,
    item_id: int,
    owner_user_id: int,
    require_delivery: Callable[[], None] | None,
) -> int | None:
    path = resolve_vn_asset_storage_path(user_id=owner_user_id, storage_path=storage_ref)
    if path != DatabasePaths.get_user_outputs_dir(owner_user_id).resolve() / storage_ref:
        raise ImportCleanupConflict("import_cleanup_storage_path_mismatch")

    def require_owned_file(record: dict[str, Any]) -> None:
        if not generated_file_matches_vn_asset(record, user_id=owner_user_id, item_id=item_id):
            raise ImportCleanupConflict("import_cleanup_file_mismatch")
        if repo.count_items_referencing_generated_file(int(record["id"]), exclude_item_id=item_id):
            raise ImportCleanupConflict("import_cleanup_shared_file")

    def remove_bytes() -> None:
        path.unlink(missing_ok=True)
        remove_stale_vn_import_parts(path)

    file_id = await storage.discard_vn_import_file(
        user_id=owner_user_id,
        source_ref=vn_asset_source_ref(item_id),
        storage_path=storage_ref,
        require_owned_file=require_owned_file,
        remove_bytes=remove_bytes,
        require_delivery=require_delivery,
    )
    logger.debug("Discarded staged VN import asset {} for user {}", item_id, owner_user_id)
    return file_id
