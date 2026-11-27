"""Cleanup blocker integration for VN generated assets."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNPlay_DB import VNPlayRepository
from tldw_Server_API.app.core.DB_Management.VNScripts_DB import VNScriptsRepository


class VNAssetCleanupBlockerProvider:
    """Find generated-file references that prevent physical asset cleanup."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self.scripts_repo = VNScriptsRepository(db)
        self.play_repo = VNPlayRepository.initialized(db)

    def find_blockers(
        self,
        *,
        pack_id: int | None,
        owner_user_id: int,
        candidates: list[dict[str, Any]],
    ) -> dict[int, list[dict[str, str]]]:
        """Return blockers keyed by file ID, across the owner's packs when pack_id is absent."""
        file_ids = {
            int(candidate["generated_file_id"])
            for candidate in candidates
            if candidate.get("generated_file_id") is not None
        }
        item_ids_by_file_id = {
            int(candidate["generated_file_id"]): int(candidate["id"])
            for candidate in candidates
            if candidate.get("generated_file_id") is not None
        }
        if not file_ids:
            return {}

        blockers = self.scripts_repo.find_asset_cleanup_blockers(
            owner_user_id=owner_user_id,
            asset_pack_id=pack_id,
            generated_file_ids=file_ids,
        )
        play_blockers = self.play_repo.find_asset_cleanup_blockers(
            owner_user_id=owner_user_id,
            asset_pack_id=pack_id,
            generated_file_ids=file_ids,
            item_ids_by_file_id=item_ids_by_file_id,
        )
        for file_id, entries in play_blockers.items():
            blockers.setdefault(file_id, []).extend(entries)
        return blockers

    def find_runtime_storage_paths(self, *, owner_user_id: int, storage_paths: set[str]) -> set[str]:
        """Keep raw bytes referenced even when their file registration has been removed."""
        if not storage_paths:
            return set()
        paths = storage_paths | {path.replace("/", "\\") for path in storage_paths}
        refs = self.scripts_repo.find_published_storage_paths(owner_user_id=owner_user_id, storage_paths=paths)
        refs.update(self.play_repo.find_runtime_storage_paths(owner_user_id=owner_user_id, storage_paths=paths))
        return {ref.replace("\\", "/") for ref in refs}
