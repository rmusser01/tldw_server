"""Journaled VN pack import commit executor."""

from __future__ import annotations

import base64
import json
import zipfile
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from inspect import isawaitable
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.VN_Assets.portability.constants import (
    ASSET_BYTES_STATUS_MISSING,
    ASSET_BYTES_STATUS_PRESENT,
    CHECKSUMS_PATH,
    MANIFEST_PATH,
    TRUST_MODE_TRUSTED_RESTORE,
    TRUST_MODE_UNTRUSTED_IMPORT,
)
from tldw_Server_API.app.core.VN_Assets.portability.conflicts import build_update_existing_plan
from tldw_Server_API.app.core.VN_Assets.portability.fingerprints import sha256_file
from tldw_Server_API.app.core.VN_Assets.portability.preview import (
    VNPackImportPreviewer,
    _archive_members_by_normalized_name,
    _read_optional_json,
    _read_required_json,
    _section_list,
    _section_record,
)
from tldw_Server_API.app.core.VN_Assets.portability.recovery import (
    discard_import_checkpoint,
    legacy_import_may_have_written,
)
from tldw_Server_API.app.core.VN_Assets.storage import generated_file_matches_vn_asset, image_format_from_mime_type

QuotaPreflight = Callable[..., tuple[bool, dict[str, Any]] | Awaitable[tuple[bool, dict[str, Any]]]]


class VNPackImporter:
    """Commit a validated VN pack preview into local VN metadata and storage."""

    def __init__(
        self,
        *,
        repo: VNAssetPacksRepository,
        owner_user_id: int,
        save_vn_asset_image: Callable[..., dict[str, Any] | Awaitable[dict[str, Any]]],
        unregister_generated_file: Callable[..., bool | Awaitable[bool]] | None = None,
        preflight_storage_quota: QuotaPreflight | None = None,
        import_storage: Any = None,
        require_delivery: Callable[[], None] | None = None,
    ) -> None:
        self.repo = repo
        self.owner_user_id = int(owner_user_id)
        self.save_vn_asset_image = save_vn_asset_image
        self.unregister_generated_file = unregister_generated_file
        self.preflight_storage_quota = preflight_storage_quota
        self.import_storage = import_storage
        self.require_delivery = require_delivery

    async def import_pack(
        self,
        *,
        preview_id: int,
        job_id: str,
        trust_mode: str,
        target_mode: str,
        character_action: str,
        target_character_id: int | None = None,
        target_pack_id: int | None = None,
        conflict_decisions: Mapping[str, Any] | None = None,
        journal_id: int | None = None,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if target_mode not in {"create_new", "update_existing"}:
            raise ValueError("unsupported_import_target_mode")
        if trust_mode not in {TRUST_MODE_TRUSTED_RESTORE, TRUST_MODE_UNTRUSTED_IMPORT}:
            raise ValueError("unsupported_import_trust_mode")
        if self.require_delivery is not None:
            self.require_delivery()

        admitted_at = None
        if journal_id is not None:
            current = self.repo.get_import_journal(journal_id, owner_user_id=self.owner_user_id)
            if (
                current is None or int(current["preview_id"]) != preview_id or current["job_id"] != job_id
                or current["target_mode"] != target_mode or current["trust_mode"] != trust_mode
            ):
                raise ValueError("import_journal_not_found")
            cleanup = json.loads(current["cleanup_status_json"])
            legacy_completion = (
                current["status"] == "failed"
                and target_mode == "create_new"
                and current["completed_at"] is not None
                and cleanup.get("status") == "not_required"
            )
            if current["status"] == "completed" or legacy_completion:
                pack_id = int(current["target_pack_id"] or 0)
                if pack_id <= 0:
                    raise ValueError("vn_pack_import_completed_result_missing")
                if legacy_completion:
                    self.repo.update_import_journal(
                        journal_id,
                        {"status": "completed", "stage": "completed", "error_code": None, "error_message": None},
                        owner_user_id=self.owner_user_id,
                    )
                return {
                    "status": "imported", "import_id": journal_id, "preview_id": preview_id,
                    "pack_id": pack_id, "id_maps": json.loads(current["id_maps_json"]),
                    "created_records": json.loads(current["created_records_json"]),
                }
            if cleanup.get("status") == "discarding" and not cleanup.get("abandoned"):
                await discard_import_checkpoint(
                    self.repo, import_id=journal_id, owner_user_id=self.owner_user_id, abandoned=False,
                    storage=self.import_storage, unregister=self.unregister_generated_file,
                    require_delivery=self.require_delivery,
                )
            elif cleanup.get("status") in {"discarding", "discarded"}:
                raise ValueError("import_cleanup_required")
            admitted_at = _parse_datetime(current["created_at"])

        preview = self._load_preview(preview_id)
        archive_path = Path(str(preview["archive_path"]))
        archive_sha256 = str(preview.get("archive_sha256") or "")
        canonical_payload_fingerprint = str(preview.get("canonical_payload_fingerprint") or "")
        self._validate_preview_ready(preview=preview, archive_path=archive_path, admitted_at=admitted_at)
        self._progress(progress, "revalidating_preview", {"preview_id": int(preview_id)})
        await self._revalidate_archive(
            archive_path=archive_path,
            expected_archive_sha256=archive_sha256,
            expected_fingerprint=canonical_payload_fingerprint,
        )

        journal = self._ensure_journal(
            journal_id=journal_id,
            preview_id=preview_id,
            job_id=job_id,
            trust_mode=trust_mode,
            target_mode=target_mode,
            archive_path=archive_path,
            archive_sha256=archive_sha256,
            canonical_payload_fingerprint=canonical_payload_fingerprint,
        )
        import_id = int(journal["id"])
        try:
            archive_data = _load_archive_payloads(archive_path)
            pack_id, id_maps, created_records = await self._prepare_import(
                journal=journal,
                preview=preview,
                archive_data=archive_data,
                character_action=character_action,
                target_character_id=target_character_id,
                target_pack_id=target_pack_id,
                conflict_decisions=conflict_decisions or {},
            )
            await self._store_prepared_assets(
                import_id=import_id,
                archive_path=archive_path,
                pack_id=pack_id,
                items=archive_data["items"],
                id_maps=id_maps,
                created_records=created_records,
                progress=progress,
            )
            return self._complete_import(
                journal=journal,
                preview_id=preview_id,
                pack_id=pack_id,
                items=archive_data["items"],
                id_maps=id_maps,
                trust_mode=trust_mode,
                progress=progress,
            )
        except Exception as exc:
            if self.require_delivery is not None:
                self.require_delivery()
            current = self.repo.get_import_journal(import_id, owner_user_id=self.owner_user_id) or journal
            if current["status"] == "completed":
                raise
            try:
                await discard_import_checkpoint(
                    self.repo, import_id=import_id, owner_user_id=self.owner_user_id, abandoned=False,
                    storage=self.import_storage, unregister=self.unregister_generated_file,
                    require_delivery=self.require_delivery,
                )
            except Exception as cleanup_error:
                logger.warning("VN import {} cleanup requires attention ({})", import_id, type(cleanup_error).__name__)
            if self.require_delivery is not None:
                self.require_delivery()
            self.repo.update_import_journal(
                import_id,
                {
                    "status": "failed",
                    "stage": "failed",
                    "error_code": "import_failed",
                    "error_message": str(exc),
                },
                owner_user_id=self.owner_user_id,
            )
            raise

    async def _prepare_import(
        self,
        *,
        journal: dict[str, Any],
        preview: dict[str, Any],
        archive_data: dict[str, Any],
        character_action: str,
        target_character_id: int | None,
        target_pack_id: int | None,
        conflict_decisions: Mapping[str, Any],
    ) -> tuple[int, dict[str, Any], dict[str, Any]]:
        """Commit the journal and hidden local rows before writing asset files."""
        checkpoint = self._resume_checkpoint(journal)
        plan_args = {
            "archive_data": archive_data,
            "accepted_plan": _preview_update_existing_plan(preview),
            "target_pack_id": target_pack_id,
            "target_character_id": target_character_id,
            "conflict_decisions": conflict_decisions,
        }
        updating = journal["target_mode"] == "update_existing"
        items = archive_data["items"]
        if checkpoint is None:
            if updating:
                candidate = self._validated_update_candidate(**plan_args)
                added = {int(entry["source_item_id"]) for entry in candidate["added_items"]}
                await self._preflight_quota([item for item in items if int(item["source_item_id"]) in added])
            else:
                await self._preflight_quota(items)

        import_id = int(journal["id"])
        with self.repo.db.transaction():
            current = self._require_processing_import(import_id)
            checkpoint = self._resume_checkpoint(current)
            if checkpoint is not None:
                pack_id = int(checkpoint["pack_id"])
                self._require_owned_pack(pack_id)
                if updating:
                    if pack_id != target_pack_id:
                        raise ValueError("import_target_changed")
                    self._validated_update_candidate(**plan_args)
                return pack_id, json.loads(current["id_maps_json"]), checkpoint

            id_maps: dict[str, Any] = {"packs": {}, "slots": {}, "items": {}, "characters": {}}
            created_records: dict[str, Any] = {
                "checkpoint_version": 1,
                "writes_started": True,
                "slot_ids": [],
                "item_ids": [],
                "generated_file_ids": [],
                "asset_formats": {},
            }
            if updating:
                candidate = self._validated_update_candidate(**plan_args)
                pack_id = int(candidate["target_pack_id"])
                if pack_id in self.repo.import_staging_ids(pack_id=pack_id)["targets"]:
                    raise ValueError("import_target_busy")
                slot_by_source_id = self._map_existing_slots(candidate=candidate, id_maps=id_maps)
                self._create_update_slots(
                    pack_id=pack_id,
                    slots=archive_data["slots"],
                    candidate=candidate,
                    slot_by_source_id=slot_by_source_id,
                    id_maps=id_maps,
                    created_records=created_records,
                )
                for matched in candidate["matched_items"]:
                    id_maps["items"][str(matched["source_item_id"])] = int(matched["local_item_id"])
                added = {int(entry["source_item_id"]) for entry in candidate["added_items"]}
                items = [item for item in items if int(item["source_item_id"]) in added]
            else:
                previous_character = json.loads(current["created_records_json"]).get("new_character_id")
                character_id = self._resolve_character(
                    action="link_existing_character" if previous_character else character_action,
                    target_character_id=previous_character or target_character_id,
                    pack=archive_data["pack"], character=archive_data.get("character"), id_maps=id_maps,
                )
                if character_action != "link_existing_character":
                    created_records["new_character_id"] = character_id
                pack_id = int(self._create_pack(archive_data["pack"], primary_character_id=character_id)["id"])
                slot_by_source_id = self._create_slots(
                    pack_id=pack_id,
                    slots=archive_data["slots"],
                    id_maps=id_maps,
                    created_records=created_records,
                )
            created_records["pack_id"] = pack_id
            source_pack_id = archive_data["pack"].get("source_pack_id")
            if source_pack_id not in (None, ""):
                id_maps["packs"][str(source_pack_id)] = pack_id
            self._prepare_items(
                import_id=import_id,
                pack_id=pack_id,
                slots=archive_data["slots"],
                items=items,
                slot_by_source_id=slot_by_source_id,
                id_maps=id_maps,
                created_records=created_records,
            )
            self.repo.update_import_journal(
                import_id,
                {"stage": "creating_items", "id_maps": id_maps, "created_records": created_records, "cleanup_status": {}},
                owner_user_id=self.owner_user_id,
            )
        return pack_id, id_maps, created_records

    def _resume_checkpoint(self, journal: Mapping[str, Any]) -> dict[str, Any] | None:
        cleanup = json.loads(journal["cleanup_status_json"]).get("status")
        if cleanup in {"partial", "discarding", "discarded"}:
            raise ValueError("import_cleanup_required")
        records = json.loads(journal["created_records_json"])
        if records.get("checkpoint_version") == 1:
            return None if cleanup == "completed" else records
        if legacy_import_may_have_written(records):
            raise ValueError("import_legacy_checkpoint_incomplete")
        return None

    def _require_owned_pack(self, pack_id: int) -> dict[str, Any]:
        pack = self.repo.get_pack(pack_id)
        if pack is None or bool(pack["deleted"]) or int(pack["owner_user_id"]) != self.owner_user_id:
            raise ValueError("target_pack_not_found")
        return pack

    def _require_processing_import(self, import_id: int) -> dict[str, Any]:
        if self.require_delivery is not None:
            self.require_delivery()
        current = self.repo.get_import_journal(import_id, owner_user_id=self.owner_user_id)
        if current is None or current["status"] != "processing":
            raise ValueError("import_journal_not_processing")
        if json.loads(current["cleanup_status_json"]).get("status") in {"discarding", "discarded"}:
            raise ValueError("import_cleanup_required")
        return current

    def _load_preview(self, preview_id: int) -> dict[str, Any]:
        preview = self.repo.get_import_preview(preview_id, owner_user_id=self.owner_user_id)
        if preview is None:
            raise ValueError("import_preview_not_found")
        return preview

    def _validate_preview_ready(
        self, *, preview: Mapping[str, Any], archive_path: Path, admitted_at: datetime | None,
    ) -> None:
        if str(preview.get("status") or "") != "completed":
            raise ValueError("import_preview_not_completed")
        if not archive_path.is_file():
            raise ValueError("import_archive_not_found")
        expires_at = _parse_datetime(preview.get("expires_at"))
        if (
            expires_at is not None and expires_at <= datetime.now(timezone.utc)
            and (admitted_at is None or admitted_at >= expires_at)
        ):
            raise ValueError("import_preview_expired")
        if sha256_file(archive_path) != str(preview.get("archive_sha256") or ""):
            raise ValueError("import_archive_checksum_changed")

    async def _revalidate_archive(
        self,
        *,
        archive_path: Path,
        expected_archive_sha256: str,
        expected_fingerprint: str,
    ) -> None:
        result = await VNPackImportPreviewer(repo=self.repo).create_preview(
            archive_path=archive_path,
            owner_user_id=self.owner_user_id,
        )
        if result["archive_sha256"] != expected_archive_sha256:
            raise ValueError("import_archive_checksum_changed")
        if result["canonical_payload_fingerprint"] != expected_fingerprint:
            raise ValueError("import_payload_fingerprint_changed")

    def _ensure_journal(
        self,
        *,
        journal_id: int | None,
        preview_id: int,
        job_id: str,
        trust_mode: str,
        target_mode: str,
        archive_path: Path,
        archive_sha256: str,
        canonical_payload_fingerprint: str,
    ) -> dict[str, Any]:
        if journal_id is not None:
            journal = self.repo.get_import_journal(journal_id, owner_user_id=self.owner_user_id)
            if journal is None or int(journal["preview_id"]) != int(preview_id):
                raise ValueError("import_journal_not_found")
            self.repo.update_import_journal(
                int(journal["id"]),
                {
                    "status": "processing",
                    "stage": "revalidating_preview",
                    "job_id": job_id,
                    "trust_mode": trust_mode,
                    "target_mode": target_mode,
                    "archive_path": str(archive_path),
                    "archive_sha256": archive_sha256,
                    "canonical_payload_fingerprint": canonical_payload_fingerprint,
                },
                owner_user_id=self.owner_user_id,
            )
            return self.repo.get_import_journal(int(journal["id"]), owner_user_id=self.owner_user_id) or journal

        return self.repo.create_import_journal(
            owner_user_id=self.owner_user_id,
            preview_id=preview_id,
            job_id=job_id,
            status="processing",
            stage="revalidating_preview",
            trust_mode=trust_mode,
            target_mode=target_mode,
            archive_path=str(archive_path),
            archive_sha256=archive_sha256,
            canonical_payload_fingerprint=canonical_payload_fingerprint,
            created_records={"writes_started": False},
        )

    def _resolve_character(
        self,
        *,
        action: str,
        target_character_id: int | None,
        pack: Mapping[str, Any],
        character: Mapping[str, Any] | None,
        id_maps: dict[str, Any],
    ) -> int:
        source_character_id = pack.get("primary_character_id")
        if action == "link_existing_character":
            if target_character_id is None:
                raise ValueError("target_character_required")
            if self.repo.get_character(int(target_character_id)) is None:
                raise ValueError("target_character_not_found")
            if source_character_id not in (None, ""):
                id_maps["characters"][str(source_character_id)] = int(target_character_id)
            return int(target_character_id)

        if action == "import_included_character":
            if character is None:
                raise ValueError("included_character_missing")
            character_id = self.repo.db.add_character_card(_character_card_payload(character))
            if character_id is None:
                raise ValueError("character_import_failed")
            if source_character_id not in (None, ""):
                id_maps["characters"][str(source_character_id)] = int(character_id)
            return int(character_id)

        if action == "create_placeholder_character":
            name = f"Imported placeholder {source_character_id or 'character'}"
            character_id = self.repo.db.add_character_card(
                {
                    "name": name,
                    "description": "Imported placeholder character. Relink this VN pack to the original source character.",
                    "extensions": {"tldw": {"vn_pack_import_placeholder": True}},
                }
            )
            if character_id is None:
                raise ValueError("character_placeholder_failed")
            if source_character_id not in (None, ""):
                id_maps["characters"][str(source_character_id)] = int(character_id)
            return int(character_id)

        raise ValueError("primary_character_unresolved")

    def _create_pack(self, pack: Mapping[str, Any], *, primary_character_id: int) -> dict[str, Any]:
        created = self.repo.create_pack(
            owner_user_id=self.owner_user_id,
            primary_character_id=primary_character_id,
            title=str(pack.get("title") or "Imported VN Pack"),
            description=pack.get("description"),
            content_rating=str(pack.get("content_rating") or "general"),
            source_world_book_ids=[],
            scenario_notes=pack.get("scenario_notes"),
            style_prompt=pack.get("style_prompt"),
            negative_prompt=pack.get("negative_prompt"),
            default_backend=pack.get("default_backend"),
            default_model=pack.get("default_model"),
            default_dimensions=_mapping_or_none(pack.get("default_dimensions")),
            style_lock=_mapping_or_none(pack.get("style_lock")),
            generation_budget=_mapping_or_none(pack.get("generation_budget")),
        )
        return self.repo.update_pack(int(created["id"]), {"status": pack.get("status") or "draft"}) or created

    def _create_slots(
        self,
        *,
        pack_id: int,
        slots: list[dict[str, Any]],
        id_maps: dict[str, Any],
        created_records: dict[str, Any],
    ) -> dict[int, dict[str, Any]]:
        created_by_source_id: dict[int, dict[str, Any]] = {}
        pending_dependents: list[dict[str, Any]] = []
        for slot in slots:
            if slot.get("depends_on_slot_id") not in (None, ""):
                pending_dependents.append(slot)
                continue
            created = self._create_slot(pack_id=pack_id, slot=slot)
            source_slot_id = int(slot["source_slot_id"])
            created_by_source_id[source_slot_id] = created
            id_maps["slots"][str(source_slot_id)] = int(created["id"])
            created_records["slot_ids"].append(int(created["id"]))

        for slot in pending_dependents:
            source_parent_id = int(slot["depends_on_slot_id"])
            parent = created_by_source_id.get(source_parent_id)
            if parent is None:
                raise ValueError("dependent_slot_not_found")
            created = self._create_slot(pack_id=pack_id, slot=slot, depends_on_slot_id=int(parent["id"]))
            source_slot_id = int(slot["source_slot_id"])
            created_by_source_id[source_slot_id] = created
            id_maps["slots"][str(source_slot_id)] = int(created["id"])
            created_records["slot_ids"].append(int(created["id"]))
        return created_by_source_id

    def _create_slot(
        self,
        *,
        pack_id: int,
        slot: Mapping[str, Any],
        depends_on_slot_id: int | None = None,
    ) -> dict[str, Any]:
        return self.repo.create_slot(
            pack_id=pack_id,
            asset_type=str(slot["asset_type"]),
            slot_key=str(slot["slot_key"]),
            labels=_mapping_or_none(slot.get("labels")) or {},
            prompt_template=slot.get("prompt_template"),
            negative_prompt_template=slot.get("negative_prompt_template"),
            variant_count=int(slot.get("variant_count") or 1),
            width=_int_or_none(slot.get("width")),
            height=_int_or_none(slot.get("height")),
            backend_override=slot.get("backend_override"),
            model_override=slot.get("model_override"),
            seed_policy=_mapping_or_none(slot.get("seed_policy")),
            requires_review=bool(slot.get("requires_review", True)),
            required_for_runtime=bool(slot.get("required_for_runtime", True)),
            depends_on_slot_id=depends_on_slot_id,
            status=str(slot.get("status") or "planned"),
            last_error=slot.get("last_error"),
        )

    def _prepare_items(
        self,
        *,
        import_id: int,
        pack_id: int,
        slots: list[dict[str, Any]],
        items: list[dict[str, Any]],
        slot_by_source_id: dict[int, dict[str, Any]],
        id_maps: dict[str, Any],
        created_records: dict[str, Any],
    ) -> None:
        slot_by_identity = {
            (str(slot["asset_type"]), str(slot["slot_key"])): slot_by_source_id[int(slot["source_slot_id"])]
            for slot in slots
            if slot.get("source_slot_id") not in (None, "")
        }
        for item in sorted(items, key=lambda row: int(row.get("variant_index") or 0)):
            source_item_id = int(item["source_item_id"])
            slot = _resolve_item_slot(item=item, slot_by_source_id=slot_by_source_id, slot_by_identity=slot_by_identity)
            created_item = self.repo.create_item(
                pack_id=pack_id,
                slot_id=int(slot["id"]),
                variant_index=int(item.get("variant_index") or 0),
                file_artifact_id=item.get("file_artifact_id"),
                generated_file_id=None,
                storage_ref=None,
                mime_type=item.get("mime_type"),
                width=_int_or_none(item.get("width")),
                height=_int_or_none(item.get("height")),
                bytes=_int_or_none(item.get("bytes")),
                review_status="hidden",
                preferred=False,
                source="imported",
                generation_job_id=item.get("generation_job_id"),
                source_context_snapshot={
                    "vnpack_import": {
                        "import_id": import_id,
                        "source_item_id": source_item_id,
                        "source_review_status": item.get("review_status"),
                        "asset_bytes_status": item.get("asset_bytes_status"),
                    }
                },
                backend_metadata=_mapping_or_none(item.get("backend_metadata")),
                depth_kind=item.get("depth_kind"),
                has_alpha=None if item.get("has_alpha") is None else bool(item.get("has_alpha")),
                crop_box=_mapping_or_none(item.get("crop_box")),
                anchor=_mapping_or_none(item.get("anchor")),
                scale_hint=item.get("scale_hint"),
                trim_status=str(item.get("trim_status") or "unknown"),
                quality_flags=_list_or_none(item.get("quality_flags")) or [],
            )
            created_records["item_ids"].append(int(created_item["id"]))
            id_maps["items"][str(source_item_id)] = int(created_item["id"])
            if item.get("asset_bytes_status") == ASSET_BYTES_STATUS_PRESENT:
                item_id = int(created_item["id"])
                created_records["asset_formats"][str(item_id)] = _image_format(item)


    async def _store_prepared_assets(
        self,
        *,
        import_id: int,
        archive_path: Path,
        pack_id: int,
        items: list[dict[str, Any]],
        id_maps: dict[str, Any],
        created_records: dict[str, Any],
        progress: Callable[[str, dict[str, Any]], None] | None,
    ) -> None:
        owned_item_ids = set(created_records["item_ids"])
        saved_assets = 0
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = _archive_members_by_normalized_name(archive)
            for item in sorted(items, key=lambda row: int(row.get("variant_index") or 0)):
                source_item_id = int(item["source_item_id"])
                item_id = id_maps["items"].get(str(source_item_id))
                if item_id not in owned_item_ids or item.get("asset_bytes_status") != ASSET_BYTES_STATUS_PRESENT:
                    continue
                self._require_processing_import(import_id)
                prepared = self._require_prepared_item(import_id, pack_id, item_id, source_item_id)
                asset_bytes = archive.read(members[str(item["asset_path"])])
                slot = self.repo.get_slot(int(prepared["slot_id"]))
                if slot is None or int(slot["pack_id"]) != pack_id:
                    raise ValueError("import_checkpoint_slot_missing")
                file_record = await _maybe_await(
                    self.save_vn_asset_image(
                        user_id=self.owner_user_id,
                        image_bytes=asset_bytes,
                        image_format=_image_format(item),
                        pack_id=pack_id,
                        item_id=item_id,
                        import_id=import_id,
                        publication_guard=lambda: self._require_processing_import(import_id),
                        asset_type=str(item["asset_type"]),
                        labels=_json_mapping_or_none(slot.get("labels_json"))
                        or _mapping_or_none(item.get("labels"))
                        or {},
                        check_quota=True,
                    )
                )
                if (
                    not generated_file_matches_vn_asset(file_record, user_id=self.owner_user_id, item_id=item_id)
                    or int(file_record["file_size_bytes"]) != len(asset_bytes)
                    or file_record.get("checksum") not in {None, item.get("asset_sha256")}
                ):
                    raise ValueError("import_storage_binding_mismatch")
                file_id = int(file_record["id"])
                with self.repo.db.transaction():
                    current = self._require_processing_import(import_id)
                    attached = self._require_prepared_item(import_id, pack_id, item_id, source_item_id)
                    self.repo.update_item_storage(
                        item_id,
                        generated_file_id=file_id,
                        storage_ref=str(file_record.get("storage_path") or ""),
                        mime_type=str(file_record.get("mime_type") or item.get("mime_type") or "image/png"),
                        width=_int_or_none(item.get("width")),
                        height=_int_or_none(item.get("height")),
                        bytes=len(asset_bytes),
                        backend_metadata={
                            "vnpack_import": {
                                "source_item_id": source_item_id,
                                "source_asset_sha256": item.get("asset_sha256"),
                            }
                        },
                    )
                    records = json.loads(current["created_records_json"])
                    old_file_id = attached["generated_file_id"]
                    if old_file_id is not None and old_file_id != file_id:
                        records["generated_file_ids"] = [
                            identifier for identifier in records["generated_file_ids"] if identifier != old_file_id
                        ]
                    if file_id not in records["generated_file_ids"]:
                        records["generated_file_ids"].append(file_id)
                    self.repo.update_import_journal(
                        import_id, {"created_records": records}, owner_user_id=self.owner_user_id,
                    )
                    created_records["generated_file_ids"] = records["generated_file_ids"]
                saved_assets += 1
                self._progress(progress, "creating_items", {"created_items": saved_assets})

    def _require_prepared_item(
        self, import_id: int, pack_id: int, item_id: int, source_item_id: int
    ) -> dict[str, Any]:
        item = self.repo.get_item(item_id)
        if item is not None and int(item["pack_id"]) == pack_id:
            context = _json_mapping_or_none(item.get("source_context_snapshot_json")) or {}
            binding = context.get("vnpack_import") or {}
            if binding.get("import_id") == import_id and binding.get("source_item_id") == source_item_id:
                return item
        raise ValueError("import_checkpoint_item_missing")

    def _validated_update_candidate(
        self,
        *,
        archive_data: Mapping[str, Any],
        accepted_plan: Mapping[str, Any],
        target_pack_id: int | None,
        target_character_id: int | None,
        conflict_decisions: Mapping[str, Any],
    ) -> dict[str, Any]:
        if target_pack_id is None:
            raise ValueError("target_pack_required")
        target_pack = self._require_owned_pack(int(target_pack_id))
        if target_character_id is not None and int(target_pack["primary_character_id"]) != int(target_character_id):
            raise ValueError("update_character_relink_not_supported")

        plan = build_update_existing_plan(
            repo=self.repo,
            owner_user_id=self.owner_user_id,
            manifest=archive_data["manifest"],
            pack=archive_data["pack"],
            slots=archive_data["slots"],
            items=archive_data["items"],
        )
        candidate = _candidate_for_target(plan, int(target_pack_id))
        if candidate is None:
            raise ValueError("update_target_not_found")
        accepted_candidate = _candidate_for_target(accepted_plan, int(target_pack_id))
        if accepted_candidate is None:
            raise ValueError("update_preview_candidate_not_found")
        if _candidate_signature(candidate) != _candidate_signature(accepted_candidate):
            raise ValueError("update_plan_changed")
        blocking_diffs = [
            diff for diff in candidate["diffs"] if str(diff.get("severity")) == "blocking"
        ]
        if blocking_diffs:
            raise ValueError("update_blocked")
        manual_resolution_diffs = [
            diff for diff in candidate["diffs"] if str(diff.get("kind")) == "item_variant_index_ambiguous"
        ]
        if manual_resolution_diffs:
            raise ValueError("update_manual_resolution_required")
        confirmed_diff_ids = set(conflict_decisions.get("confirm_diff_ids") or [])
        confirm_all = bool(conflict_decisions.get("confirm_all_risky_diffs"))
        unconfirmed = [
            diff
            for diff in candidate["diffs"]
            if bool(diff.get("requires_confirmation"))
            and not confirm_all
            and str(diff["diff_id"]) not in confirmed_diff_ids
        ]
        if unconfirmed:
            raise ValueError("update_confirmation_required")

        return candidate

    def _map_existing_slots(
        self,
        *,
        candidate: Mapping[str, Any],
        id_maps: dict[str, Any],
    ) -> dict[int, dict[str, Any]]:
        mapped: dict[int, dict[str, Any]] = {}
        for matched_slot in candidate["matched_slots"]:
            source_slot_id = int(matched_slot["source_slot_id"])
            local_slot_id = int(matched_slot["local_slot_id"])
            local_slot = self.repo.get_slot(local_slot_id)
            if local_slot is None:
                raise ValueError("update_slot_not_found")
            mapped[source_slot_id] = local_slot
            id_maps["slots"][str(source_slot_id)] = local_slot_id
        return mapped

    def _create_update_slots(
        self,
        *,
        pack_id: int,
        slots: list[dict[str, Any]],
        candidate: Mapping[str, Any],
        slot_by_source_id: dict[int, dict[str, Any]],
        id_maps: dict[str, Any],
        created_records: dict[str, Any],
    ) -> dict[int, dict[str, Any]]:
        added_source_slot_ids = {
            int(slot["source_slot_id"])
            for slot in candidate["added_slots"]
            if slot.get("source_slot_id") is not None
        }
        created: dict[int, dict[str, Any]] = {}
        remaining = [slot for slot in slots if int(slot["source_slot_id"]) in added_source_slot_ids]
        while remaining:
            next_remaining: list[dict[str, Any]] = []
            created_this_pass = False
            for slot in remaining:
                depends_on_slot_id = None
                if slot.get("depends_on_slot_id") not in (None, ""):
                    parent = slot_by_source_id.get(int(slot["depends_on_slot_id"]))
                    if parent is None:
                        next_remaining.append(slot)
                        continue
                    depends_on_slot_id = int(parent["id"])
                created_slot = self._create_slot(
                    pack_id=pack_id,
                    slot=slot,
                    depends_on_slot_id=depends_on_slot_id,
                )
                source_slot_id = int(slot["source_slot_id"])
                slot_by_source_id[source_slot_id] = created_slot
                created[source_slot_id] = created_slot
                id_maps["slots"][str(source_slot_id)] = int(created_slot["id"])
                created_records["slot_ids"].append(int(created_slot["id"]))
                created_this_pass = True
            if not created_this_pass:
                raise ValueError("dependent_slot_not_found")
            remaining = next_remaining
        return created

    def _complete_import(
        self,
        *,
        journal: Mapping[str, Any],
        preview_id: int,
        pack_id: int,
        items: list[dict[str, Any]],
        id_maps: dict[str, Any],
        trust_mode: str,
        progress: Callable[[str, dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        import_id = int(journal["id"])
        with self.repo.db.transaction():
            current = self._require_processing_import(import_id)
            created_records = json.loads(current["created_records_json"])
            owned_item_ids = set(created_records["item_ids"])
            protected_preferences = {
                int(item["slot_id"]) for item in self.repo.list_items(pack_id, include_import_staging=True)
                if item["preferred"] and int(item["id"]) not in owned_item_ids
            }
            for item in sorted(items, key=lambda row: int(row.get("variant_index") or 0)):
                source_item_id = int(item["source_item_id"])
                item_id = id_maps["items"].get(str(source_item_id))
                if item_id not in owned_item_ids:
                    continue
                prepared = self._require_prepared_item(import_id, pack_id, item_id, source_item_id)
                if item.get("asset_bytes_status") == ASSET_BYTES_STATUS_PRESENT and not prepared["generated_file_id"]:
                    raise ValueError("import_checkpoint_file_missing")
                review_status, preferred = _final_review_state(item, trust_mode=trust_mode)
                self.repo.update_item_review(
                    item_id,
                    review_status=review_status,
                    preferred=preferred and int(prepared["slot_id"]) not in protected_preferences,
                )
            self.repo.update_import_journal(
                import_id,
                {
                    "status": "completed",
                    "stage": "completed",
                    "target_pack_id": pack_id,
                    "id_maps": id_maps,
                    "created_records": created_records,
                    "cleanup_status": {"status": "not_required"},
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                owner_user_id=self.owner_user_id,
            )
        self._progress(
            progress,
            "completed",
            {"pack_id": pack_id, "generated_file_count": len(created_records["generated_file_ids"])},
        )
        return {
            "status": "imported",
            "import_id": import_id,
            "preview_id": int(preview_id),
            "pack_id": pack_id,
            "id_maps": id_maps,
            "created_records": created_records,
        }

    def _progress(
        self,
        progress: Callable[[str, dict[str, Any]], None] | None,
        stage: str,
        payload: dict[str, Any],
    ) -> None:
        if progress is not None:
            progress(stage, payload)

    async def _preflight_quota(self, items: list[Mapping[str, Any]]) -> None:
        if self.preflight_storage_quota is None:
            return
        total_asset_bytes = sum(
            _int_or_none(item.get("asset_size_bytes")) or _int_or_none(item.get("bytes")) or 0
            for item in items
            if item.get("asset_bytes_status") == ASSET_BYTES_STATUS_PRESENT
        )
        if total_asset_bytes <= 0:
            return
        await _maybe_await(
            self.preflight_storage_quota(
                self.owner_user_id,
                total_asset_bytes,
                raise_on_exceed=True,
            )
        )


def _load_archive_payloads(archive_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = _archive_members_by_normalized_name(archive)
        # Ensure checksum metadata is present before commit parsing. Preview already validates it.
        _read_required_json(archive, members, CHECKSUMS_PATH)
        manifest = _read_required_json(archive, members, MANIFEST_PATH)
        pack_payload = _read_required_json(archive, members, "metadata/pack.json")
        slots_payload = _read_required_json(archive, members, "metadata/slots.json")
        items_payload = _read_required_json(archive, members, "metadata/items.json")
        character_payload = _read_optional_json(archive, members, "metadata/character.json")
    return {
        "manifest": manifest,
        "pack": _section_record(pack_payload, key="pack", path="metadata/pack.json"),
        "slots": _section_list(slots_payload, key="slots", path="metadata/slots.json"),
        "items": _section_list(items_payload, key="items", path="metadata/items.json"),
        "character": (
            _section_record(character_payload, key="character", path="metadata/character.json")
            if character_payload is not None
            else None
        ),
    }


def _preview_update_existing_plan(preview: Mapping[str, Any]) -> Mapping[str, Any]:
    proposed_plan = _json_mapping_or_none(preview.get("proposed_plan_json")) or {}
    update_existing = proposed_plan.get("update_existing")
    return update_existing if isinstance(update_existing, Mapping) else {}


def _resolve_item_slot(
    *,
    item: Mapping[str, Any],
    slot_by_source_id: Mapping[int, Mapping[str, Any]],
    slot_by_identity: Mapping[tuple[str, str], Mapping[str, Any]],
) -> Mapping[str, Any]:
    if item.get("source_slot_id") not in (None, ""):
        slot = slot_by_source_id.get(int(item["source_slot_id"]))
        if slot is not None:
            return slot
    slot = slot_by_identity.get((str(item.get("asset_type")), str(item.get("slot_key"))))
    if slot is None:
        raise ValueError("import_item_slot_not_found")
    return slot


def _candidate_for_target(plan: Mapping[str, Any], target_pack_id: int) -> Mapping[str, Any] | None:
    for candidate in plan.get("candidate_packs", []):
        if int(candidate["target_pack_id"]) == int(target_pack_id):
            return candidate
    return None


def _candidate_signature(candidate: Mapping[str, Any]) -> dict[str, tuple[tuple[Any, ...], ...]]:
    return {
        "matched_slots": tuple(
            sorted(
                (
                    _signature_int(slot.get("source_slot_id")),
                    _signature_int(slot.get("local_slot_id")),
                    str(slot.get("identity") or ""),
                )
                for slot in candidate.get("matched_slots", [])
            )
        ),
        "added_slots": tuple(
            sorted(
                (
                    _signature_int(slot.get("source_slot_id")),
                    str(slot.get("identity") or ""),
                )
                for slot in candidate.get("added_slots", [])
            )
        ),
        "matched_items": tuple(
            sorted(
                (
                    _signature_int(item.get("source_item_id")),
                    _signature_int(item.get("local_item_id")),
                    _signature_int(item.get("source_slot_id")),
                    _signature_int(item.get("local_slot_id")),
                    str(item.get("match_kind") or ""),
                )
                for item in candidate.get("matched_items", [])
            )
        ),
        "added_items": tuple(
            sorted(
                (
                    _signature_int(item.get("source_item_id")),
                    _signature_int(item.get("source_slot_id")),
                )
                for item in candidate.get("added_items", [])
            )
        ),
        "diffs": tuple(
            sorted(
                (
                    str(diff.get("diff_id") or ""),
                    str(diff.get("kind") or ""),
                    str(diff.get("severity") or ""),
                    bool(diff.get("requires_confirmation")),
                )
                for diff in candidate.get("diffs", [])
            )
        ),
    }


def _signature_int(value: Any) -> int:
    return -1 if value in (None, "") else int(value)


def _final_review_state(item: Mapping[str, Any], *, trust_mode: str) -> tuple[str, bool]:
    if item.get("asset_bytes_status") == ASSET_BYTES_STATUS_MISSING:
        return "hidden", False
    if trust_mode == TRUST_MODE_TRUSTED_RESTORE:
        review_status = str(item.get("review_status") or "draft")
        preferred = bool(item.get("preferred")) if review_status == "approved" else False
        return review_status, preferred
    return "draft", False


def _image_format(item: Mapping[str, Any]) -> str:
    mime_type = str(item.get("mime_type") or "").strip()
    if mime_type:
        return image_format_from_mime_type(mime_type)
    asset_path = str(item.get("asset_path") or "")
    suffix = Path(asset_path).suffix.lstrip(".").lower()
    if suffix == "jpeg":
        return "jpg"
    return suffix or "png"


def _character_card_payload(character: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        key: character.get(key)
        for key in (
            "name",
            "description",
            "personality",
            "scenario",
            "system_prompt",
            "post_history_instructions",
            "first_message",
            "message_example",
            "creator_notes",
            "alternate_greetings",
            "tags",
            "creator",
            "character_version",
            "extensions",
        )
    }
    image_base64 = character.get("image_base64")
    if image_base64:
        payload["image"] = base64.b64decode(str(image_base64))
    return {key: value for key, value in payload.items() if value is not None}


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _json_mapping_or_none(value: Any) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    if isinstance(value, Mapping):
        return dict(value)
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return None
    return dict(loaded) if isinstance(loaded, Mapping) else None


def _list_or_none(value: Any) -> list[Any] | None:
    return list(value) if isinstance(value, list) else None


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


async def _maybe_await(value: Any) -> Any:
    if isawaitable(value):
        return await value
    return value
