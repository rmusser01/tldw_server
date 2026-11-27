"""Stored asset checks used by VN manifest reads and approval."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

from tldw_Server_API.app.core.VN_Assets.storage import (
    VN_ASSET_CONTENT_NOT_FOUND,
    generated_file_size_bytes,
    resolve_registered_vn_asset_content,
)

_FILE_LOOKUP_BATCH_SIZE = 300
VNAssetRuntimeFileLoader = Callable[[list[int]], Awaitable[list[dict[str, Any]]]]


@dataclass(frozen=True)
class VNAssetRuntimeReference:
    """Item identity and the file metadata needed for an integrity check."""

    item_id: int
    file_id: int | None
    mime_type: str | None


class VNAssetRuntimeIssue(TypedDict):
    """A stable failure code for one item in an owned pack."""

    item_id: int
    code: str


def manifest_runtime_references(assets: Mapping[str, list[dict[str, Any]]]) -> list[VNAssetRuntimeReference]:
    """Collect file references from the v1 entries selected by the manifest builder."""
    return [
        VNAssetRuntimeReference(asset["item_id"], asset["generated_file_id"], asset["mime_type"])
        for collection in assets.values()
        for asset in collection
    ]


async def inspect_runtime_assets(
    references: Sequence[VNAssetRuntimeReference],
    *,
    user_id: int,
    load_file_records: VNAssetRuntimeFileLoader,
) -> list[VNAssetRuntimeIssue]:
    """Load registrations in bounded batches and check local files off-thread."""
    if not references:
        return []

    file_ids = list(dict.fromkeys(ref.file_id for ref in references if ref.file_id is not None))
    records: dict[int, dict[str, Any]] = {}
    for offset in range(0, len(file_ids), _FILE_LOOKUP_BATCH_SIZE):
        batch = await load_file_records(file_ids[offset:offset + _FILE_LOOKUP_BATCH_SIZE])
        records.update((int(record["id"]), record) for record in batch)
    return await asyncio.to_thread(_inspect_local_files, references, records, user_id)


def _inspect_local_files(
    references: Sequence[VNAssetRuntimeReference],
    records: dict[int, dict[str, Any]],
    user_id: int,
) -> list[VNAssetRuntimeIssue]:
    issues: list[VNAssetRuntimeIssue] = []
    for ref in references:
        record = records.get(ref.file_id) if ref.file_id is not None else None
        try:
            path, _ = resolve_registered_vn_asset_content(
                record, user_id=user_id, item_id=ref.item_id, fallback_mime_type=ref.mime_type,
            )
            # lgtm[py/path-injection]: path comes from the owner-local content resolver.
            actual_size = path.stat().st_size
        except OSError:
            issues.append({"item_id": ref.item_id, "code": VN_ASSET_CONTENT_NOT_FOUND})
        except ValueError as exc:
            issues.append({"item_id": ref.item_id, "code": str(exc)})
        else:
            expected_size = generated_file_size_bytes(record or {})
            if actual_size <= 0 or (expected_size > 0 and actual_size != expected_size):
                issues.append({"item_id": ref.item_id, "code": "vn_asset_content_size_mismatch"})
    return issues
