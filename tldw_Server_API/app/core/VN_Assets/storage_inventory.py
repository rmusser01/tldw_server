"""Read-only retention and storage reconciliation for owner-local VN assets."""

from __future__ import annotations

import asyncio
import os
import re
import stat
from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import (
    AuthnzGeneratedFilesRepo,
    SOURCE_FEATURE_VN_ASSETS,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.VN_Assets.cleanup_blockers import VNAssetCleanupBlockerProvider
from tldw_Server_API.app.core.VN_Assets.constants import ITEM_REVIEW_STATUS_HIDDEN, ITEM_REVIEW_STATUS_REJECTED
from tldw_Server_API.app.core.VN_Assets.jobs import VN_ASSETS_DOMAIN, VN_PACK_EXPORT_JOB_TYPE, VN_PACK_IMPORT_COMMIT_JOB_TYPE
from tldw_Server_API.app.core.VN_Assets.storage import generated_file_matches_vn_asset, generated_file_size_bytes
from tldw_Server_API.app.core.VN_Assets.vn_asset_schemas import (
    VNAssetStorageFinding,
    VNAssetStorageOrphanReport,
    VNStorageScanSource,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Jobs.manager import JobManager

VN_RETENTION_DAYS = {ITEM_REVIEW_STATUS_REJECTED: 30, ITEM_REVIEW_STATUS_HIDDEN: 7}
_ORPHAN_GRACE = timedelta(hours=24)
_VN_SOURCE = re.compile(r"vn_asset_item:([1-9][0-9]{0,18})")
_VN_FILENAME = re.compile(r"vn_asset_([1-9][0-9]{0,18})_[0-9]{8}_[0-9]{6}_[0-9a-f]{12}\.[a-z0-9]+")
_MAX_ITEM_ID = 2**63 - 1


def retention_has_elapsed(item: Mapping[str, Any], *, now: datetime) -> bool:
    """Retain all assets outside the age policy or without a trustworthy timestamp."""
    days = VN_RETENTION_DAYS.get(item["review_status"])
    return days is not None and _old_enough(item, now - timedelta(days=days))


def _old_enough(row: Mapping[str, Any], cutoff: datetime) -> bool:
    value = row.get("updated_at") or row.get("created_at")
    if value is None:
        return False
    try:
        updated = datetime.fromisoformat(str(value))
    except ValueError:
        return False
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    return updated <= cutoff


async def storage_work_is_active(
    repo: VNAssetPacksRepository, *, owner_user_id: int, pack_id: int | None = None, jobs_manager: JobManager | None = None,
) -> bool:
    """Consult local publication state and the owner's live portability queue."""
    def read_activity() -> dict[str, bool]:
        active = repo.storage_work_is_active(
            owner_user_id=owner_user_id, pack_id=pack_id, include_bound_portability=jobs_manager is None,
        )
        if not active and jobs_manager is not None:
            active = _live_portability_work(jobs_manager, owner_user_id=owner_user_id, pack_id=pack_id)
        return {"active": active}

    result = await repo.run_worker_replay_operation(read_activity)
    return bool(result and result["active"])


def _live_portability_work(jobs_manager: JobManager, *, owner_user_id: int, pack_id: int | None) -> bool:
    limit = 1 if pack_id is None else 100
    for job_type, target_key in (
        (VN_PACK_EXPORT_JOB_TYPE, "pack_id"),
        (VN_PACK_IMPORT_COMMIT_JOB_TYPE, "target_pack_id"),
    ):
        for status in ("queued", "processing"):
            jobs = jobs_manager.list_jobs(
                domain=VN_ASSETS_DOMAIN, owner_user_id=str(owner_user_id), job_type=job_type, status=status, limit=limit,
            )
            # Preserve safety when the whole active queue cannot fit in one lookup.
            if len(jobs) >= limit:
                return True
            for job in jobs:
                payload = job.get("payload")
                target_pack_id = payload.get(target_key) if isinstance(payload, Mapping) else None
                if target_pack_id is None or str(target_pack_id) == str(pack_id):
                    return True
    return False


async def inspect_storage_orphans(
    repo: VNAssetPacksRepository,
    files_repo: AuthnzGeneratedFilesRepo,
    blocker_provider: VNAssetCleanupBlockerProvider,
    *,
    owner_user_id: int,
    source: VNStorageScanSource,
    offset: int,
    limit: int,
    jobs_manager: JobManager | None = None,
) -> VNAssetStorageOrphanReport:
    """Inspect one bounded page without reading image contents or changing storage."""
    if await storage_work_is_active(repo, owner_user_id=owner_user_id, jobs_manager=jobs_manager):
        return VNAssetStorageOrphanReport(source=source, scan_deferred=True)

    cutoff = datetime.now(timezone.utc) - _ORPHAN_GRACE
    if source == "items":
        page = await repo.run_worker_replay_operation(
            lambda: {"items": repo.list_storage_items(owner_user_id=owner_user_id, offset=offset, limit=limit + 1)},
        )
        rows = page["items"] if page else []
        findings = await _missing_registrations(files_repo, rows[:limit], owner_user_id=owner_user_id, cutoff=cutoff)
        has_more = len(rows) > limit
    else:
        outputs = await asyncio.to_thread(_owner_outputs, owner_user_id)
        if source == "registrations":
            rows, total = await files_repo.list_files(
                user_id=owner_user_id, source_feature=SOURCE_FEATURE_VN_ASSETS, offset=offset, limit=limit,
            )
            page = await repo.run_worker_replay_operation(
                lambda: {"findings": _registration_findings(
                    repo, blocker_provider, rows, outputs=outputs, owner_user_id=owner_user_id, cutoff=cutoff,
                )},
            )
            findings = page["findings"] if page else []
            has_more = offset + len(rows) < total
        else:
            rows = await asyncio.to_thread(_managed_file_page, outputs, offset, limit + 1)
            findings = await _unregistered_files(
                repo, files_repo, blocker_provider, rows[:limit], owner_user_id=owner_user_id, cutoff=cutoff.timestamp(),
            )
            has_more = len(rows) > limit

    if await storage_work_is_active(repo, owner_user_id=owner_user_id, jobs_manager=jobs_manager):
        return VNAssetStorageOrphanReport(source=source, scan_deferred=True)
    return VNAssetStorageOrphanReport(
        source=source, findings=findings, scanned_count=min(limit, len(rows)),
        next_offset=offset + limit if has_more else None,
    )


def _owner_outputs(owner_user_id: int) -> Path:
    return DatabasePaths.get_user_outputs_dir(owner_user_id).resolve()


def _vn_storage_ref(value: Any) -> str | None:
    if not isinstance(value, str) or "\x00" in value:
        return None
    normalized = value.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if len(relative.parts) < 2 or relative.parts[0] != "vn_assets" or ".." in relative.parts:
        return None
    return normalized if relative.as_posix() == normalized else None


async def _missing_registrations(
    files_repo: AuthnzGeneratedFilesRepo, rows: list[dict[str, Any]], *, owner_user_id: int, cutoff: datetime,
) -> list[VNAssetStorageFinding]:
    items = [item for item in rows if _old_enough(item, cutoff)]
    records = {record["id"]: record for record in await files_repo.get_files_by_ids(
        sorted({int(item["generated_file_id"]) for item in items}),
    )}
    findings = []
    for item in items:
        file_id = int(item["generated_file_id"])
        item_id = int(item["id"])
        record = records.get(file_id)
        if record and generated_file_matches_vn_asset(
            record, user_id=owner_user_id, item_id=item_id, include_deleted=True,
        ):
            continue
        findings.append(VNAssetStorageFinding(
            kind="missing_registration", item_id=item_id, pack_id=item["pack_id"],
            generated_file_id=file_id, storage_ref=_vn_storage_ref(item["storage_ref"]),
        ))
    return findings


def _regular_file_at(outputs: Path, storage_ref: str) -> tuple[bool, os.stat_result | None]:
    """Distinguish missing managed bytes from a path we cannot safely inspect."""
    # lgtm[py/path-injection] The caller validates a relative VN path; reject symlink aliases before stat.
    path = outputs / storage_ref
    try:
        if path.resolve() != path:
            return False, None
        info = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return True, None
    except (NotADirectoryError, RuntimeError):
        return False, None
    return stat.S_ISREG(info.st_mode), info


def _registration_findings(
    repo: VNAssetPacksRepository,
    blocker_provider: VNAssetCleanupBlockerProvider,
    rows: list[dict[str, Any]],
    *,
    outputs: Path,
    owner_user_id: int,
    cutoff: datetime,
) -> list[VNAssetStorageFinding]:
    findings = []
    unreferenced = []
    for record in rows:
        source = _VN_SOURCE.fullmatch(str(record.get("source_ref") or ""))
        ref = _vn_storage_ref(record.get("storage_path"))
        if source is None or ref is None or not _old_enough(record, cutoff):
            continue
        item_id = int(source[1])
        if item_id > _MAX_ITEM_ID or not generated_file_matches_vn_asset(record, user_id=owner_user_id, item_id=item_id):
            continue
        valid_path, info = _regular_file_at(outputs, ref)
        if not valid_path:
            continue
        file_id = int(record["id"])
        if info is None:
            findings.append(VNAssetStorageFinding(
                kind="missing_bytes", item_id=item_id, generated_file_id=file_id, storage_ref=ref,
                bytes=generated_file_size_bytes(record),
            ))
        elif info.st_mtime <= cutoff.timestamp() and not repo.storage_reference_exists(
            owner_user_id=owner_user_id, item_id=item_id, storage_ref=ref, generated_file_id=file_id,
        ):
            unreferenced.append(VNAssetStorageFinding(
                kind="unreferenced_registration", item_id=item_id, generated_file_id=file_id, storage_ref=ref,
                bytes=generated_file_size_bytes(record),
            ))
    if unreferenced:
        blockers = blocker_provider.find_blockers(
            pack_id=None, owner_user_id=owner_user_id,
            candidates=[{"id": row.item_id, "generated_file_id": row.generated_file_id} for row in unreferenced],
        )
        runtime_paths = blocker_provider.find_runtime_storage_paths(
            owner_user_id=owner_user_id,
            storage_paths={row.storage_ref for row in unreferenced if row.storage_ref is not None},
        )
        findings.extend(
            row for row in unreferenced
            if row.generated_file_id not in blockers and row.storage_ref not in runtime_paths
        )
    return findings


def _managed_file_page(outputs: Path, offset: int, limit: int) -> list[tuple[int, str, os.stat_result]]:
    return list(islice(_managed_files(outputs), offset, offset + limit))


def _managed_files(outputs: Path) -> Iterator[tuple[int, str, os.stat_result]]:
    root = outputs / "vn_assets"
    if root.is_symlink() or not root.exists():
        return

    def raise_walk_error(exc: OSError) -> None:
        raise exc

    for directory, subdirs, filenames in os.walk(root, onerror=raise_walk_error, followlinks=False):
        subdirs.sort()
        for filename in sorted(filenames):
            match = _VN_FILENAME.fullmatch(filename)
            if match is None or (item_id := int(match[1])) > _MAX_ITEM_ID:
                continue
            ref = (Path(directory) / filename).relative_to(outputs).as_posix()
            valid_path, info = _regular_file_at(outputs, ref)
            if valid_path and info is not None:
                yield item_id, ref, info


async def _unregistered_files(
    repo: VNAssetPacksRepository,
    files_repo: AuthnzGeneratedFilesRepo,
    blocker_provider: VNAssetCleanupBlockerProvider,
    files: list[tuple[int, str, os.stat_result]],
    *,
    owner_user_id: int,
    cutoff: float,
) -> list[VNAssetStorageFinding]:
    old_files = [(item_id, ref, info) for item_id, ref, info in files if info.st_mtime <= cutoff]
    registered = await files_repo.registered_storage_paths(
        user_id=owner_user_id,
        storage_paths=[path for _, ref, _ in old_files for path in (ref, ref.replace("/", "\\"))],
    )
    registered = {path.replace("\\", "/") for path in registered}

    def find_unregistered_files() -> dict[str, Any]:
        candidates = [
            (item_id, ref, info) for item_id, ref, info in old_files
            if ref not in registered and not repo.storage_reference_exists(
                owner_user_id=owner_user_id, item_id=item_id, storage_ref=ref,
            )
        ]
        known_paths = blocker_provider.find_runtime_storage_paths(
            owner_user_id=owner_user_id, storage_paths={ref for _, ref, _ in candidates},
        )
        return {"findings": [
            VNAssetStorageFinding(kind="unregistered_bytes", item_id=item_id, storage_ref=ref, bytes=info.st_size)
            for item_id, ref, info in candidates if ref not in known_paths
        ]}

    result = await repo.run_worker_replay_operation(find_unregistered_files)
    return result["findings"] if result else []
