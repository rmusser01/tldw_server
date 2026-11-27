"""Cleanup audit snapshots stored alongside the owner's idempotency receipt."""

from __future__ import annotations

import json
from dataclasses import dataclass

from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.VN_Assets.vn_asset_schemas import (
    VNAssetCleanupAuditEntry,
    VNAssetCleanupFile,
    VNAssetCleanupHistoryResponse,
    VNAssetCleanupRequest,
    VNAssetCleanupResponse,
    VNAssetCleanupSelection,
)

MAX_AUDIT_ITEM_IDS = 300


def cleanup_audit_selection(request: VNAssetCleanupRequest) -> VNAssetCleanupSelection:
    """Capture the effective filters with a bounded list of valid item IDs."""
    requested_ids = request.item_ids or []
    item_ids = [item_id for item_id in requested_ids[:MAX_AUDIT_ITEM_IDS] if 1 <= item_id <= 2**63 - 1]
    return VNAssetCleanupSelection(
        statuses=list(dict.fromkeys(request.effective_statuses)),
        item_ids=item_ids if requested_ids else None,
        include_approved=request.include_approved,
        item_ids_truncated=len(item_ids) < len(requested_ids),
    )


@dataclass(frozen=True)
class CleanupAudit:
    """Persist the public result and private audit metadata for one admitted run."""

    repo: VNAssetPacksRepository
    owner_user_id: int
    pack_id: int
    audit_id: int
    selection: VNAssetCleanupSelection

    async def save(
        self,
        response: VNAssetCleanupResponse,
        active_file: VNAssetCleanupFile | None,
        *,
        complete: bool = False,
    ) -> None:
        """Commit the latest confirmed effects before proceeding with cleanup."""
        response.audit_id = self.audit_id
        snapshot = response.model_dump(mode="json")
        snapshot["_cleanup_audit"] = {
            "selection": self.selection.model_dump(mode="json"),
            "active_file": active_file.model_dump(mode="json") if active_file else None,
        }
        await self.repo.run_worker_replay_operation(
            lambda: self.repo.save_cleanup_checkpoint(
                audit_id=self.audit_id,
                owner_user_id=self.owner_user_id,
                pack_id=self.pack_id,
                snapshot=snapshot,
                complete=complete,
            ),
        )


def cleanup_history(
    repo: VNAssetPacksRepository,
    *,
    owner_user_id: int,
    pack_id: int,
    before_id: int | None,
    limit: int,
) -> VNAssetCleanupHistoryResponse:
    """Project receipt data for an owned pack, including a soft-deleted pack."""
    pack = repo.get_pack(pack_id)
    if pack is None or int(pack["owner_user_id"]) != owner_user_id:
        raise ValueError("pack_not_found")

    rows = repo.list_cleanup_receipts(
        owner_user_id=owner_user_id, pack_id=pack_id, before_id=before_id, limit=limit + 1,
    )
    entries = []
    for row in rows[:limit]:
        audit_id = int(row["id"])
        snapshot = json.loads(row["response_json"])
        metadata = snapshot.pop("_cleanup_audit", {})
        result = VNAssetCleanupResponse.model_validate({**snapshot, "audit_id": audit_id}) if snapshot else None
        if row["status"] == "in_progress":
            status = "in_progress"
        else:
            status = "partial_failure" if result and result.failed_step else "completed"
        entries.append(
            VNAssetCleanupAuditEntry(
                id=audit_id,
                owner_user_id=owner_user_id,
                pack_id=pack_id,
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                status=status,
                selection=metadata.get("selection"),
                result=result,
                active_file=metadata.get("active_file"),
            )
        )
    return VNAssetCleanupHistoryResponse(
        entries=entries,
        next_before_id=entries[-1].id if len(rows) > limit else None,
    )
