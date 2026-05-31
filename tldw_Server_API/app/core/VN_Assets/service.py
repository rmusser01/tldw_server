"""Service orchestration for VN asset pack metadata."""

from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
from collections.abc import Mapping
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
    VALID_REVIEW_STATUSES,
    VNAssetBulkReviewRequest,
    VNAssetCleanupRequest,
    VNAssetCleanupResponse,
    VNAssetGenerationRequest,
    VNAssetGenerationStatusResponse,
    VNAssetItemResponse,
    VNAssetManifestResponse,
    VNAssetPackCreate,
    VNAssetPackResponse,
    VNAssetPackUpdate,
    VNAssetPromptPreviewRequest,
    VNAssetPromptPreviewResponse,
    VNAssetReadinessResponse,
    VNAssetReviewRequest,
    VNAssetSlotCreate,
    VNAssetSlotResponse,
    VNAssetSlotUpdate,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.Storage import generated_file_helpers
from tldw_Server_API.app.core.VN_Assets.constants import (
    ASSET_TYPE_BACKGROUND,
    ASSET_TYPE_DEPTH_COMPANION,
    DEFAULT_VN_ASSET_PACK_ITEM_LIMIT,
    DEFAULT_VN_ASSET_SLOT_VARIANT_LIMIT,
    ERROR_ITEM_LIMIT_EXCEEDED,
    ERROR_SLOT_VARIANT_LIMIT_EXCEEDED,
    ITEM_REVIEW_STATUS_APPROVED,
    SLOT_STATUS_CANCELLED,
    SLOT_STATUS_FAILED,
    SLOT_STATUS_SKIPPED,
    WARNING_DEPTH_UNAVAILABLE,
)
from tldw_Server_API.app.core.VN_Assets.jobs import create_enqueue_batch_job
from tldw_Server_API.app.core.VN_Assets.manifest import build_manifest as build_core_manifest
from tldw_Server_API.app.core.VN_Assets.matrix import expand_starter_matrix
from tldw_Server_API.app.core.VN_Assets.models import (
    SlotReadiness,
    VNAssetItem,
    VNAssetPack,
    VNAssetSlot,
)
from tldw_Server_API.app.core.VN_Assets.prompts import PromptBudgets, build_prompt_preview
from tldw_Server_API.app.core.VN_Assets.state import derive_pack_readiness, derive_slot_status
from tldw_Server_API.app.core.VN_Assets.storage import (
    detect_image_dimensions,
    generated_file_matches_vn_asset,
    generated_file_size_bytes,
    image_format_from_mime_type,
    resolve_vn_asset_storage_path,
    unlink_vn_asset_storage_file,
)

VN_ASSET_APPROVED_CLEANUP_CONFIRMATION = "DELETE APPROVED VN ASSETS"


class VNAssetPackService:
    """Pure service/repository orchestration for VN asset packs."""

    def __init__(
        self,
        db: CharactersRAGDB,
        *,
        owner_user_id: int,
        item_limit: int = DEFAULT_VN_ASSET_PACK_ITEM_LIMIT,
        jobs_manager: Any | None = None,
    ) -> None:
        self.repo = VNAssetPacksRepository.initialized(db)
        self.owner_user_id = owner_user_id
        self.item_limit = item_limit
        self.jobs_manager = jobs_manager

    def create_pack(self, request: VNAssetPackCreate) -> VNAssetPackResponse:
        requested_planned_count = _planned_output_count_from_budget(request.generation_budget)
        if requested_planned_count is not None:
            self._enforce_item_limit(requested_planned_count)

        if request.apply_starter_matrix:
            if self.repo.get_character(request.primary_character_id) is None:
                raise ValueError("primary_character_not_found")
            expand_starter_matrix(
                primary_character_id=request.primary_character_id,
                variant_count=request.starter_matrix_variant_count,
                max_items=self.item_limit,
            )

        pack = self.repo.create_pack(
            owner_user_id=self.owner_user_id,
            primary_character_id=request.primary_character_id,
            title=request.title,
            description=request.description,
            content_rating=request.content_rating,
            source_world_book_ids=request.source_world_book_ids,
            scenario_notes=request.scenario_notes,
            style_prompt=request.style_prompt,
            negative_prompt=request.negative_prompt,
            default_backend=request.default_backend,
            default_model=request.default_model,
            default_dimensions=request.default_dimensions,
            style_lock=request.style_lock,
            generation_budget=request.generation_budget,
        )
        response = self._pack_response(pack)
        if request.apply_starter_matrix:
            self.apply_matrix(
                response.id,
                "starter",
                {"variant_count": request.starter_matrix_variant_count},
            )
            return self.get_pack(response.id)
        return response

    def list_packs(self) -> list[VNAssetPackResponse]:
        return [
            self._pack_response(pack)
            for pack in self.repo.list_packs(owner_user_id=self.owner_user_id)
        ]

    def list_packs_for_setup(
        self,
        *,
        query: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[list[VNAssetPackResponse], bool]:
        rows, has_more = self.repo.list_packs_for_setup(
            owner_user_id=self.owner_user_id,
            query=query,
            limit=limit,
            offset=offset,
        )
        return [
            self._pack_response(pack, include_planned_output_count=False)
            for pack in rows
        ], has_more

    def get_pack(self, pack_id: int) -> VNAssetPackResponse:
        return self._pack_response(self._require_pack(pack_id))

    def update_pack(self, pack_id: int, request: VNAssetPackUpdate) -> VNAssetPackResponse:
        self._require_pack(pack_id)
        fields = request.model_dump(exclude_unset=True)
        requested_planned_count = _planned_output_count_from_budget(fields.get("generation_budget"))
        if requested_planned_count is not None:
            self._enforce_item_limit(requested_planned_count)
        updated = self.repo.update_pack(pack_id, fields)
        if updated is None:
            raise ValueError("pack_not_found")
        self._enforce_pack_item_limit(pack_id)
        return self._pack_response(updated)

    def soft_delete_pack(self, pack_id: int) -> None:
        self._require_pack(pack_id)
        self.repo.soft_delete_pack(pack_id)

    def apply_matrix(
        self,
        pack_id: int,
        matrix_key: str,
        overrides: dict[str, Any],
    ) -> list[VNAssetSlotResponse]:
        pack = self._require_pack(pack_id)
        if matrix_key != "starter":
            raise ValueError("unknown_matrix")

        variant_count = _matrix_variant_count_from_overrides(overrides)
        if variant_count > DEFAULT_VN_ASSET_SLOT_VARIANT_LIMIT:
            raise ValueError(ERROR_SLOT_VARIANT_LIMIT_EXCEEDED)
        matrix_slots = expand_starter_matrix(
            primary_character_id=int(pack["primary_character_id"]),
            variant_count=variant_count,
            max_items=self.item_limit,
        )
        existing_total = self._planned_output_count(pack_id)
        matrix_total = sum(slot.variant_count for slot in matrix_slots)
        self._enforce_item_limit(existing_total + matrix_total)
        created_rows = self.repo.create_slots_for_matrix(
            pack_id=pack_id,
            slot_specs=[_slot_create_spec(slot) for slot in matrix_slots],
        )
        return [self._slot_response(row) for row in created_rows]

    def list_slots(self, pack_id: int) -> list[VNAssetSlotResponse]:
        self._require_pack(pack_id)
        return [self._slot_response(row) for row in self.repo.list_slots(pack_id)]

    def create_slot(self, pack_id: int, request: VNAssetSlotCreate) -> VNAssetSlotResponse:
        self._require_pack(pack_id)
        if request.depends_on_slot_id is not None:
            self._require_slot_in_pack(pack_id, request.depends_on_slot_id)
        self._enforce_item_limit(self._planned_output_count(pack_id) + request.variant_count)
        try:
            created = self.repo.create_slot(
                pack_id=pack_id,
                asset_type=request.asset_type,
                slot_key=request.slot_key,
                labels=request.labels,
                prompt_template=request.prompt_template,
                negative_prompt_template=request.negative_prompt_template,
                variant_count=request.variant_count,
                width=request.width,
                height=request.height,
                backend_override=request.backend_override,
                model_override=request.model_override,
                seed_policy=request.seed_policy,
                requires_review=request.requires_review,
                required_for_runtime=request.required_for_runtime,
                depends_on_slot_id=request.depends_on_slot_id,
                status=request.status,
                last_error=request.last_error,
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("slot_already_exists") from exc
        return self._slot_response(created)

    def update_slot_for_pack(
        self,
        pack_id: int,
        slot_id: int,
        request: VNAssetSlotUpdate,
    ) -> VNAssetSlotResponse:
        self._require_slot_in_pack(pack_id, slot_id)
        if request.depends_on_slot_id is not None:
            self._validate_slot_dependency(pack_id, slot_id, request.depends_on_slot_id)
        return self.update_slot(slot_id, request)

    def update_slot(self, slot_id: int, request: VNAssetSlotUpdate) -> VNAssetSlotResponse:
        slot = self._require_owned_slot(slot_id)
        fields = request.model_dump(exclude_unset=True)
        pack_id = int(slot["pack_id"])
        if "depends_on_slot_id" in fields and fields["depends_on_slot_id"] is not None:
            self._validate_slot_dependency(
                pack_id,
                slot_id,
                int(fields["depends_on_slot_id"]),
            )
        if "slot_key" in fields and any(
            int(row["id"]) != slot_id and str(row["slot_key"]) == str(fields["slot_key"])
            for row in self.repo.list_slots(pack_id)
        ):
            raise ValueError("slot_already_exists")
        if "variant_count" in fields:
            current_total = self._planned_output_count(pack_id)
            new_total = current_total - int(slot["variant_count"]) + int(fields["variant_count"])
            self._enforce_item_limit(new_total)
        try:
            updated = self.repo.update_slot(slot_id, fields)
        except sqlite3.IntegrityError as exc:
            raise ValueError("slot_already_exists") from exc
        if updated is None:
            raise ValueError("slot_not_found")
        return self._slot_response(updated)

    def delete_slot(self, pack_id: int, slot_id: int) -> None:
        self._require_slot_in_pack(pack_id, slot_id)
        if any(
            int(row["depends_on_slot_id"] or 0) == slot_id
            for row in self.repo.list_slots(pack_id)
        ):
            raise ValueError("slot_has_dependents")
        try:
            self.repo.delete_slot(slot_id)
        except sqlite3.IntegrityError as exc:
            raise ValueError("slot_has_dependents") from exc

    def list_items(self, pack_id: int) -> list[VNAssetItemResponse]:
        self._require_pack(pack_id)
        return [self._item_response(row) for row in self.repo.list_items(pack_id)]

    def get_item_for_pack(self, pack_id: int, item_id: int) -> VNAssetItemResponse:
        return self._item_response(self._require_item_in_pack(pack_id, item_id))

    async def cleanup_pack(
        self,
        pack_id: int,
        request: VNAssetCleanupRequest,
        *,
        files_repo: Any,
        unregister_generated_file: Any | None = None,
        blocker_provider: Any | None = None,
    ) -> VNAssetCleanupResponse:
        self._require_pack(pack_id)
        statuses = _cleanup_statuses(request)
        if invalid_statuses := sorted(statuses - VALID_REVIEW_STATUSES):
            raise ValueError(f"invalid_review_status:{','.join(invalid_statuses)}")

        includes_approved = ITEM_REVIEW_STATUS_APPROVED in statuses
        if includes_approved and not request.include_approved:
            raise ValueError("cleanup_approved_requires_include_approved")
        if includes_approved and not _approved_cleanup_confirmed(request):
            raise ValueError("cleanup_confirmation_required")

        item_ids = set(request.item_ids or [])
        skipped_file_ids: list[int] = []
        candidate_items: list[dict[str, Any]] = []

        for item in self.repo.list_items(pack_id):
            item_id = int(item["id"])
            if item_ids and item_id not in item_ids:
                continue
            if str(item["review_status"]) not in statuses:
                continue
            raw_file_id = item["generated_file_id"]
            if raw_file_id is None:
                continue
            file_id = int(raw_file_id)
            if self.repo.count_items_referencing_generated_file(file_id, exclude_item_id=item_id) > 0:
                skipped_file_ids.append(file_id)
                continue
            candidate_items.append(item)

        blockers_by_file_id = await _cleanup_blockers_for_items(
            blocker_provider,
            pack_id=pack_id,
            owner_user_id=self.owner_user_id,
            candidates=candidate_items,
        )
        cleanup_blocked: list[dict[str, Any]] = []
        if blockers_by_file_id:
            unblocked_items: list[dict[str, Any]] = []
            for item in candidate_items:
                file_id = int(item["generated_file_id"])
                blockers = blockers_by_file_id.get(file_id)
                if blockers:
                    skipped_file_ids.append(file_id)
                    cleanup_blocked.append(
                        {
                            "item_id": int(item["id"]),
                            "file_id": file_id,
                            "blockers": blockers,
                        }
                    )
                    continue
                unblocked_items.append(item)
            candidate_items = unblocked_items

        candidate_results = await asyncio.gather(
            *(
                _cleanup_candidate_for_item(
                    item,
                    files_repo=files_repo,
                    owner_user_id=self.owner_user_id,
                )
                for item in candidate_items
            )
        )
        candidates = [
            candidate
            for candidate in candidate_results
            if candidate is not None
        ]
        candidate_reclaimed_bytes = sum(byte_count for _, _, byte_count in candidates)

        files_would_delete = len(candidates)
        if request.dry_run:
            return VNAssetCleanupResponse(
                dry_run=True,
                removed_item_ids=[],
                removed_file_count=0,
                files_would_delete=files_would_delete,
                files_deleted=0,
                skipped_file_ids=skipped_file_ids,
                blocked_count=len(cleanup_blocked),
                cleanup_blocked=cleanup_blocked,
                reclaimed_bytes=candidate_reclaimed_bytes,
            )

        removed_item_ids: list[int] = []
        files_deleted = 0
        reclaimed_bytes = 0
        for item, record, byte_count in candidates:
            file_id = int(record["id"])
            storage_path = str(record.get("storage_path") or "")
            if storage_path:
                try:
                    resolve_vn_asset_storage_path(
                        user_id=self.owner_user_id,
                        storage_path=storage_path,
                    )
                except ValueError:
                    logger.warning(
                        "Skipping VN asset cleanup with invalid storage path: pack_id={} item_id={} file_id={}",
                        pack_id,
                        item["id"],
                        file_id,
                    )
                    skipped_file_ids.append(file_id)
                    continue

            if not await _hard_delete_generated_file(
                files_repo,
                file_id,
                unregister_generated_file=unregister_generated_file,
            ):
                skipped_file_ids.append(file_id)
                continue

            item_id = int(item["id"])
            self.repo.delete_item(item_id)
            if storage_path:
                await asyncio.to_thread(
                    unlink_vn_asset_storage_file,
                    user_id=self.owner_user_id,
                    storage_path=storage_path,
                )

            removed_item_ids.append(item_id)
            files_deleted += 1
            reclaimed_bytes += byte_count

        return VNAssetCleanupResponse(
            dry_run=False,
            removed_item_ids=removed_item_ids,
            removed_file_count=files_deleted,
            files_would_delete=files_would_delete,
            files_deleted=files_deleted,
            skipped_file_ids=skipped_file_ids,
            blocked_count=len(cleanup_blocked),
            cleanup_blocked=cleanup_blocked,
            reclaimed_bytes=reclaimed_bytes,
        )

    async def upload_item(
        self,
        pack_id: int,
        *,
        slot_id: int,
        image_bytes: bytes,
        mime_type: str,
        variant_index: int = 0,
    ) -> VNAssetItemResponse:
        self._require_pack(pack_id)
        slot = self._require_slot_in_pack(pack_id, slot_id)
        _validate_variant_index(slot, variant_index)
        if len(self.repo.list_items(pack_id)) + 1 > self.item_limit:
            raise ValueError(ERROR_ITEM_LIMIT_EXCEEDED)

        image_format = image_format_from_mime_type(mime_type)
        width, height = detect_image_dimensions(image_bytes, mime_type)
        item = self.repo.create_item(
            pack_id=pack_id,
            slot_id=slot_id,
            variant_index=variant_index,
            mime_type=mime_type,
            width=width,
            height=height,
            bytes=len(image_bytes),
            review_status="draft",
            source="uploaded",
        )
        item_id = int(item["id"])
        try:
            record = await generated_file_helpers.save_and_register_vn_asset_image(
                user_id=self.owner_user_id,
                image_bytes=image_bytes,
                image_format=image_format,
                pack_id=pack_id,
                item_id=item_id,
                asset_type=str(slot["asset_type"]),
                labels=_loads_json(slot["labels_json"], {}),
            )
            updated_item = self.repo.update_item_storage(
                item_id,
                generated_file_id=int(record["id"]),
                storage_ref=record.get("storage_path"),
                mime_type=str(record.get("mime_type") or mime_type),
                width=width,
                height=height,
                bytes=len(image_bytes),
                backend_metadata={
                    "upload": {
                        "original_mime_type": mime_type,
                        "image_format": image_format,
                    }
                },
            )
        except Exception:
            self.repo.delete_item(item_id)
            raise
        return self._item_response(updated_item or item)

    def review_item_for_pack(
        self,
        pack_id: int,
        item_id: int,
        request: VNAssetReviewRequest,
    ) -> VNAssetItemResponse:
        self._require_item_in_pack(pack_id, item_id)
        return self.review_item(item_id, request)

    def review_item(
        self,
        item_id: int,
        request: VNAssetReviewRequest,
    ) -> VNAssetItemResponse:
        if request.review_status not in VALID_REVIEW_STATUSES:
            raise ValueError("invalid_review_status")
        if request.preferred and request.review_status != ITEM_REVIEW_STATUS_APPROVED:
            raise ValueError("preferred_item_must_be_approved")
        self._require_owned_item(item_id)
        item = self.repo.update_item_review(
            item_id,
            review_status=request.review_status,
            preferred=request.preferred,
        )
        if item is None:
            raise ValueError("item_not_found")
        if str(item["review_status"]) == ITEM_REVIEW_STATUS_APPROVED:
            self._maybe_enqueue_lazy_depth_companion(item)
        return self._item_response(item)

    def bulk_review_items(
        self,
        request: VNAssetBulkReviewRequest,
    ) -> list[VNAssetItemResponse]:
        if request.review_status not in VALID_REVIEW_STATUSES:
            raise ValueError("invalid_review_status")
        for item_id in request.item_ids:
            self._require_owned_item(item_id)

        updated_items = self.repo.bulk_update_item_review(
            request.item_ids,
            review_status=request.review_status,
        )
        if request.review_status == ITEM_REVIEW_STATUS_APPROVED:
            for item in updated_items:
                self._maybe_enqueue_lazy_depth_companion(item)
        return [self._item_response(item) for item in updated_items]

    def bulk_review_items_for_pack(
        self,
        pack_id: int,
        request: VNAssetBulkReviewRequest,
    ) -> list[VNAssetItemResponse]:
        self._require_pack(pack_id)
        for item_id in request.item_ids:
            self._require_item_in_pack(pack_id, item_id)
        return self.bulk_review_items(request)

    def set_preferred_item(self, pack_id: int, item_id: int) -> VNAssetItemResponse:
        item = self._require_item_in_pack(pack_id, item_id)
        return self.review_item(
            item_id,
            VNAssetReviewRequest(review_status=str(item["review_status"]), preferred=True),
        )

    def get_readiness(self, pack_id: int) -> VNAssetReadinessResponse:
        self._require_pack(pack_id)
        slots = [self._slot_model(row) for row in self.repo.list_slots(pack_id)]
        items_by_slot_id = self._items_by_slot_id(pack_id)
        required: list[SlotReadiness] = []
        optional: list[SlotReadiness] = []

        for slot in slots:
            review_statuses = [
                item.review_status
                for item in items_by_slot_id.get(slot.id or 0, [])
            ]
            status = self._derived_slot_status(slot, review_statuses)
            warnings = self._slot_warnings(slot, status)
            readiness = SlotReadiness(
                slot_id=slot.id or 0,
                status=status,
                warnings=tuple(warnings),
            )
            if slot.required_for_runtime:
                required.append(readiness)
            else:
                optional.append(readiness)

        readiness = derive_pack_readiness(
            required_slots=required,
            optional_slots=optional,
            active_jobs=sum(1 for slot in slots if slot.status == "generating"),
            approved_item_errors=[],
        )
        return VNAssetReadinessResponse(
            ready=readiness.ready,
            status=readiness.status,
            warnings=list(readiness.warnings),
            errors=list(readiness.errors),
        )

    def start_generation(
        self,
        pack_id: int,
        request: VNAssetGenerationRequest | None = None,
        *,
        user_id: int | None = None,
        jobs_manager: Any | None = None,
    ) -> VNAssetGenerationStatusResponse:
        self._require_pack(pack_id)
        request = request or VNAssetGenerationRequest()
        requested_by_user_id = self.owner_user_id if user_id is None else int(user_id)
        selected_slot_ids = set(request.slot_ids)
        slots = self.repo.list_slots(pack_id)
        if selected_slot_ids:
            slots = [slot for slot in slots if int(slot["id"]) in selected_slot_ids]
            if len(slots) != len(selected_slot_ids):
                raise ValueError("slot_not_found")
        if not slots:
            raise ValueError("vn_asset_generation_no_slots")

        variant_count = request.variant_count
        total_variants = sum(int(variant_count or slot["variant_count"]) for slot in slots)
        self._enforce_item_limit(len(self.repo.list_items(pack_id)) + total_variants)

        options = dict(request.options)
        if selected_slot_ids:
            options["slot_ids"] = sorted(selected_slot_ids)
        if variant_count is not None:
            options["variant_count"] = int(variant_count)

        batch = self.repo.create_batch(
            pack_id=pack_id,
            requested_by_user_id=requested_by_user_id,
            status="queued",
            total_slots=len(slots),
            total_variants=total_variants,
            planned_count=total_variants,
            options=options,
        )
        try:
            job = create_enqueue_batch_job(
                jobs_manager or self._require_jobs_manager(),
                pack_id=pack_id,
                batch_id=int(batch["id"]),
                user_id=requested_by_user_id,
            )
        except Exception as exc:
            self.repo.update_batch(
                int(batch["id"]),
                {
                    "status": "failed",
                    "enqueue_error": str(exc),
                },
            )
            raise
        job_batch_id = str(job.get("id") or job.get("uuid") or "")
        if job_batch_id:
            batch = self.repo.update_batch(int(batch["id"]), {"job_batch_id": job_batch_id}) or batch

        return self._generation_status_response(batch)

    def get_generation_status(self, pack_id: int) -> VNAssetGenerationStatusResponse:
        self._require_pack(pack_id)
        batches = self.repo.list_batches(pack_id)
        if not batches:
            return VNAssetGenerationStatusResponse(status="idle")
        return self._generation_status_response(batches[0])

    def cancel_generation(self, pack_id: int) -> VNAssetGenerationStatusResponse:
        self._require_pack(pack_id)
        batches = self.repo.list_batches(pack_id)
        if not batches:
            return VNAssetGenerationStatusResponse(status="idle")
        batch = self.repo.update_batch(int(batches[0]["id"]), {"status": "cancelled"}) or batches[0]
        return self._generation_status_response(batch)

    def retry_slot(
        self,
        pack_id: int,
        slot_id: int,
        request: VNAssetGenerationRequest | None = None,
        *,
        user_id: int | None = None,
        jobs_manager: Any | None = None,
    ) -> VNAssetGenerationStatusResponse:
        self._require_slot_in_pack(pack_id, slot_id)
        request = request or VNAssetGenerationRequest()
        request = request.model_copy(update={"slot_ids": [slot_id]})
        return self.start_generation(pack_id, request, user_id=user_id, jobs_manager=jobs_manager)

    def regenerate_item(
        self,
        pack_id: int,
        item_id: int,
        request: VNAssetGenerationRequest | None = None,
        *,
        user_id: int | None = None,
        jobs_manager: Any | None = None,
    ) -> VNAssetGenerationStatusResponse:
        item = self._require_item_in_pack(pack_id, item_id)
        request = request or VNAssetGenerationRequest()
        request = request.model_copy(update={"slot_ids": [int(item["slot_id"])]})
        return self.start_generation(pack_id, request, user_id=user_id, jobs_manager=jobs_manager)

    def _maybe_enqueue_lazy_depth_companion(self, item: Mapping[str, Any]) -> None:
        if self.jobs_manager is None:
            return
        pack_id = int(item["pack_id"])
        slot_id = int(item["slot_id"])
        slot = self.repo.get_slot(slot_id)
        if slot is None or str(slot["asset_type"]) != ASSET_TYPE_BACKGROUND:
            return

        approved_depth_slot_ids: set[int] = set()
        for existing in self.repo.list_items(pack_id):
            if str(existing["review_status"]) != ITEM_REVIEW_STATUS_APPROVED:
                continue
            existing_slot = self.repo.get_slot(int(existing["slot_id"]))
            if existing_slot is not None and str(existing_slot["asset_type"]) == ASSET_TYPE_DEPTH_COMPANION:
                approved_depth_slot_ids.add(int(existing["slot_id"]))
        for candidate in self.repo.list_slots(pack_id):
            if str(candidate["asset_type"]) != ASSET_TYPE_DEPTH_COMPANION:
                continue
            if candidate["depends_on_slot_id"] is None or int(candidate["depends_on_slot_id"]) != slot_id:
                continue
            depth_slot_id = int(candidate["id"])
            if depth_slot_id in approved_depth_slot_ids:
                continue
            if self._has_active_depth_generation_batch(pack_id, depth_slot_id):
                continue
            try:
                self.start_generation(
                    pack_id,
                    VNAssetGenerationRequest(slot_ids=[depth_slot_id], variant_count=1),
                    user_id=self.owner_user_id,
                    jobs_manager=self.jobs_manager,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to enqueue lazy VN depth companion: pack_id={} slot_id={} depth_slot_id={} error={}",
                    pack_id,
                    slot_id,
                    depth_slot_id,
                    exc,
                )

    def _has_active_depth_generation_batch(self, pack_id: int, depth_slot_id: int) -> bool:
        active_statuses = {"queued", "enqueued", "processing"}
        for batch in self.repo.list_batches(pack_id):
            if str(batch["status"]) not in active_statuses:
                continue
            options = _loads_json(batch["options_json"], {})
            slot_ids = options.get("slot_ids") if isinstance(options, dict) else None
            if slot_ids is None:
                return True
            if not isinstance(slot_ids, list):
                continue
            if depth_slot_id in {int(slot_id) for slot_id in slot_ids}:
                return True
        return False

    def build_manifest(self, pack_id: int) -> VNAssetManifestResponse:
        pack_row = self._require_pack(pack_id)
        manifest = build_core_manifest(
            pack=self._pack_model(pack_row),
            slots=[self._slot_model(row) for row in self.repo.list_slots(pack_id)],
            items=[self._item_model(row) for row in self.repo.list_items(pack_id)],
        )
        return VNAssetManifestResponse(**manifest)

    def preview_prompt(
        self,
        pack_id: int,
        request: VNAssetPromptPreviewRequest,
    ) -> VNAssetPromptPreviewResponse:
        pack = self._require_pack(pack_id)
        slot = self._require_slot_in_pack(pack_id, request.slot_id)
        character = self.repo.get_character(int(pack["primary_character_id"]))
        if character is None:
            raise ValueError("primary_character_not_found")
        budgets = _prompt_budgets_from_request(request.budgets)
        negative_prompt = _join_prompt_parts(
            pack["negative_prompt"],
            slot["negative_prompt_template"],
        )
        preview = build_prompt_preview(
            character=character,
            pack_style=pack["style_prompt"],
            pack_scenario=pack["scenario_notes"],
            negative_prompt=negative_prompt,
            style_lock=_loads_optional_json(pack["style_lock_json"]),
            slot_template=slot["prompt_template"],
            labels=_loads_json(slot["labels_json"], {}),
            budgets=budgets,
        )
        return VNAssetPromptPreviewResponse(
            prompt=preview.prompt,
            negative_prompt=preview.negative_prompt,
            omitted_source_counts=preview.omitted_source_counts,
            token_estimates=preview.token_estimates,
            warnings=list(preview.warnings),
        )

    def _require_pack(self, pack_id: int) -> dict[str, Any]:
        pack = self.repo.get_pack(pack_id)
        if (
            pack is None
            or bool(pack["deleted"])
            or int(pack["owner_user_id"]) != self.owner_user_id
        ):
            raise ValueError("pack_not_found")
        return pack

    def _require_owned_slot(self, slot_id: int) -> dict[str, Any]:
        slot = self.repo.get_slot(slot_id)
        if slot is None:
            raise ValueError("slot_not_found")
        try:
            self._require_pack(int(slot["pack_id"]))
        except ValueError as exc:
            raise ValueError("slot_not_found") from exc
        return slot

    def _require_slot_in_pack(self, pack_id: int, slot_id: int) -> dict[str, Any]:
        self._require_pack(pack_id)
        slot = self._require_owned_slot(slot_id)
        if int(slot["pack_id"]) != pack_id:
            raise ValueError("slot_not_found")
        return slot

    def _require_owned_item(self, item_id: int) -> dict[str, Any]:
        item = self.repo.get_item(item_id)
        if item is None:
            raise ValueError("item_not_found")
        try:
            self._require_pack(int(item["pack_id"]))
        except ValueError as exc:
            raise ValueError("item_not_found") from exc
        return item

    def _require_item_in_pack(self, pack_id: int, item_id: int) -> dict[str, Any]:
        self._require_pack(pack_id)
        item = self._require_owned_item(item_id)
        if int(item["pack_id"]) != pack_id:
            raise ValueError("item_not_found")
        return item

    def _validate_slot_dependency(
        self,
        pack_id: int,
        slot_id: int,
        depends_on_slot_id: int,
    ) -> None:
        if depends_on_slot_id == slot_id:
            raise ValueError("slot_dependency_self")

        current_id: int | None = depends_on_slot_id
        seen: set[int] = set()
        while current_id is not None:
            if current_id == slot_id or current_id in seen:
                raise ValueError("slot_dependency_cycle")
            seen.add(current_id)
            current_slot = self._require_slot_in_pack(pack_id, current_id)
            current_id = current_slot["depends_on_slot_id"]

    def _pack_response(
        self,
        row: Mapping[str, Any],
        *,
        include_planned_output_count: bool = True,
    ) -> VNAssetPackResponse:
        pack_id = int(row["id"])
        return VNAssetPackResponse(
            id=pack_id,
            owner_user_id=int(row["owner_user_id"]),
            title=str(row["title"]),
            primary_character_id=int(row["primary_character_id"]),
            description=row["description"],
            status=str(row["status"]),
            content_rating=str(row["content_rating"]),
            source_world_book_ids=_loads_json(row["source_world_book_ids_json"], []),
            scenario_notes=row["scenario_notes"],
            style_prompt=row["style_prompt"],
            negative_prompt=row["negative_prompt"],
            default_backend=row["default_backend"],
            default_model=row["default_model"],
            default_dimensions=_loads_optional_json(row["default_dimensions_json"]),
            style_lock=_loads_optional_json(row["style_lock_json"]),
            generation_budget=_loads_optional_json(row["generation_budget_json"]),
            planned_output_count=(
                self._planned_output_count(pack_id)
                if include_planned_output_count
                else 0
            ),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            version=int(row["version"]),
            deleted=bool(row["deleted"]),
        )

    def _generation_status_response(self, row: Mapping[str, Any]) -> VNAssetGenerationStatusResponse:
        return VNAssetGenerationStatusResponse(
            batch_id=int(row["id"]),
            job_batch_id=row["job_batch_id"],
            status=str(row["status"]),
            total_slots=int(row["total_slots"] or 0),
            total_variants=int(row["total_variants"] or 0),
            planned_count=int(row["planned_count"] or 0),
            enqueued_count=int(row["enqueued_count"] or 0),
            completed_count=int(row["completed_count"] or 0),
            failed_count=int(row["failed_count"] or 0),
            cancelled_count=int(row["cancelled_count"] or 0),
            enqueue_error=row["enqueue_error"],
        )

    def _slot_response(self, row: Mapping[str, Any]) -> VNAssetSlotResponse:
        return VNAssetSlotResponse(
            id=int(row["id"]),
            pack_id=int(row["pack_id"]),
            asset_type=str(row["asset_type"]),
            slot_key=str(row["slot_key"]),
            labels=_loads_json(row["labels_json"], {}),
            prompt_template=row["prompt_template"],
            negative_prompt_template=row["negative_prompt_template"],
            variant_count=int(row["variant_count"]),
            width=row["width"],
            height=row["height"],
            backend_override=row["backend_override"],
            model_override=row["model_override"],
            seed_policy=_loads_optional_json(row["seed_policy_json"]),
            requires_review=bool(row["requires_review"]),
            required_for_runtime=bool(row["required_for_runtime"]),
            depends_on_slot_id=row["depends_on_slot_id"],
            status=str(row["status"]),
            last_error=row["last_error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _item_response(self, row: Mapping[str, Any]) -> VNAssetItemResponse:
        return VNAssetItemResponse(
            id=int(row["id"]),
            pack_id=int(row["pack_id"]),
            slot_id=int(row["slot_id"]),
            variant_index=int(row["variant_index"]),
            file_artifact_id=row["file_artifact_id"],
            generated_file_id=row["generated_file_id"],
            storage_ref=row["storage_ref"],
            mime_type=row["mime_type"],
            width=row["width"],
            height=row["height"],
            bytes=row["bytes"],
            review_status=str(row["review_status"]),
            preferred=bool(row["preferred"]),
            source=str(row["source"]),
            generation_job_id=row["generation_job_id"],
            depth_kind=row["depth_kind"],
            parent_item_id=row["parent_item_id"],
            has_alpha=None if row["has_alpha"] is None else bool(row["has_alpha"]),
            crop_box=_loads_optional_json(row["crop_box_json"]),
            anchor=_loads_optional_json(row["anchor_json"]),
            scale_hint=row["scale_hint"],
            trim_status=str(row["trim_status"]),
            quality_flags=_loads_json(row["quality_flags_json"], []),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _pack_model(self, row: Mapping[str, Any]) -> VNAssetPack:
        return VNAssetPack(
            id=int(row["id"]),
            owner_user_id=int(row["owner_user_id"]),
            title=str(row["title"]),
            primary_character_id=int(row["primary_character_id"]),
            description=row["description"],
            status=str(row["status"]),
            content_rating=str(row["content_rating"]),
        )

    def _slot_model(self, row: Mapping[str, Any]) -> VNAssetSlot:
        return VNAssetSlot(
            id=int(row["id"]),
            pack_id=int(row["pack_id"]),
            asset_type=str(row["asset_type"]),
            slot_key=str(row["slot_key"]),
            labels=_loads_json(row["labels_json"], {}),
            prompt_template=row["prompt_template"],
            negative_prompt_template=row["negative_prompt_template"],
            variant_count=int(row["variant_count"]),
            width=row["width"],
            height=row["height"],
            requires_review=bool(row["requires_review"]),
            required_for_runtime=bool(row["required_for_runtime"]),
            depends_on_slot_id=row["depends_on_slot_id"],
            status=str(row["status"]),
            last_error=row["last_error"],
        )

    def _item_model(self, row: Mapping[str, Any]) -> VNAssetItem:
        return VNAssetItem(
            id=int(row["id"]),
            pack_id=int(row["pack_id"]),
            slot_id=int(row["slot_id"]),
            variant_index=int(row["variant_index"]),
            review_status=str(row["review_status"]),
            generated_file_id=row["generated_file_id"],
            file_artifact_id=row["file_artifact_id"],
            storage_ref=row["storage_ref"],
            mime_type=row["mime_type"],
            width=row["width"],
            height=row["height"],
            preferred=bool(row["preferred"]),
            source=str(row["source"]),
            depth_kind=row["depth_kind"],
            parent_item_id=row["parent_item_id"],
            has_alpha=None if row["has_alpha"] is None else bool(row["has_alpha"]),
            crop_box=_loads_optional_json(row["crop_box_json"]),
            anchor=_loads_optional_json(row["anchor_json"]),
            scale_hint=row["scale_hint"],
            trim_status=str(row["trim_status"]),
            quality_flags=_loads_json(row["quality_flags_json"], []),
        )

    def _planned_output_count(self, pack_id: int) -> int:
        return sum(int(slot["variant_count"]) for slot in self.repo.list_slots(pack_id))

    def _enforce_pack_item_limit(self, pack_id: int) -> None:
        self._enforce_item_limit(self._planned_output_count(pack_id))

    def _enforce_item_limit(self, planned_output_count: int) -> None:
        if planned_output_count > self.item_limit:
            raise ValueError(ERROR_ITEM_LIMIT_EXCEEDED)

    def _require_jobs_manager(self) -> Any:
        if self.jobs_manager is not None:
            return self.jobs_manager
        from tldw_Server_API.app.core.Jobs.manager import JobManager

        self.jobs_manager = JobManager()
        return self.jobs_manager

    def _items_by_slot_id(self, pack_id: int) -> dict[int, list[VNAssetItem]]:
        grouped: dict[int, list[VNAssetItem]] = {}
        for row in self.repo.list_items(pack_id):
            item = self._item_model(row)
            grouped.setdefault(item.slot_id, []).append(item)
        return grouped

    def _derived_slot_status(self, slot: VNAssetSlot, review_statuses: list[str]) -> str:
        requested_variants = max(slot.variant_count, 1) if slot.status == SLOT_STATUS_FAILED else slot.variant_count
        return derive_slot_status(
            has_active_job=slot.status == "generating",
            has_queued_job=slot.status == "queued",
            is_skipped=slot.status == SLOT_STATUS_SKIPPED,
            is_cancelled=slot.status == SLOT_STATUS_CANCELLED,
            requested_variants=requested_variants,
            failed_variants=requested_variants if slot.status == SLOT_STATUS_FAILED else 0,
            review_statuses=review_statuses,
            required_for_runtime=slot.required_for_runtime,
        )

    def _slot_warnings(self, slot: VNAssetSlot, status: str) -> list[str]:
        warnings: list[str] = []
        if not slot.required_for_runtime and status == SLOT_STATUS_FAILED:
            warnings.append(f"optional_slot_failed:{slot.id}")
            if slot.asset_type == "depth_companion":
                warnings.append(WARNING_DEPTH_UNAVAILABLE)
        if slot.last_error and not slot.required_for_runtime:
            warnings.append(f"optional_slot_error:{slot.id}:{slot.last_error}")
        return warnings


def _planned_output_count_from_budget(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    planned = value.get("planned_output_count")
    if planned is None:
        return None
    return _strict_int_count(planned, "invalid_generation_budget")


def _cleanup_statuses(request: VNAssetCleanupRequest) -> set[str]:
    statuses = request.item_statuses if request.item_statuses is not None else request.statuses
    return {str(status) for status in statuses}


def _validate_variant_index(slot: Mapping[str, Any], variant_index: int) -> None:
    if variant_index < 0:
        raise ValueError("invalid_variant_index")
    slot_variant_count = int(slot["variant_count"] or 0)
    if slot_variant_count > 0 and variant_index >= slot_variant_count:
        raise ValueError("invalid_variant_index")


def _approved_cleanup_confirmed(request: VNAssetCleanupRequest) -> bool:
    return (
        request.confirmation_text == VN_ASSET_APPROVED_CLEANUP_CONFIRMATION
        or request.confirmation_token == VN_ASSET_APPROVED_CLEANUP_CONFIRMATION
    )


async def _get_generated_file(files_repo: Any, file_id: int) -> dict[str, Any] | None:
    get_file_by_id = getattr(files_repo, "get_file_by_id", None)
    if not callable(get_file_by_id):
        raise ValueError("generated_files_repo_unavailable")
    result = get_file_by_id(file_id)
    record = await result if inspect.isawaitable(result) else result
    return dict(record) if record is not None else None


async def _cleanup_candidate_for_item(
    item: dict[str, Any],
    *,
    files_repo: Any,
    owner_user_id: int,
) -> tuple[dict[str, Any], dict[str, Any], int] | None:
    item_id = int(item["id"])
    file_id = int(item["generated_file_id"])
    record = await _get_generated_file(files_repo, file_id)
    if not record or not generated_file_matches_vn_asset(
        record,
        user_id=owner_user_id,
        item_id=item_id,
    ):
        return None
    byte_count = generated_file_size_bytes(record, fallback=item.get("bytes"))
    return item, record, byte_count


async def _cleanup_blockers_for_items(
    blocker_provider: Any | None,
    *,
    pack_id: int,
    owner_user_id: int,
    candidates: list[dict[str, Any]],
) -> dict[int, list[dict[str, str]]]:
    if blocker_provider is None or not candidates:
        return {}
    find_blockers = getattr(blocker_provider, "find_blockers", None)
    if not callable(find_blockers):
        return {}
    result = find_blockers(
        pack_id=pack_id,
        owner_user_id=owner_user_id,
        candidates=candidates,
    )
    raw_blockers = await result if inspect.isawaitable(result) else result
    if not isinstance(raw_blockers, Mapping):
        return {}

    normalized: dict[int, list[dict[str, str]]] = {}
    for raw_file_id, blockers in raw_blockers.items():
        try:
            file_id = int(raw_file_id)
        except (TypeError, ValueError):
            continue
        if not isinstance(blockers, list):
            continue
        normalized_blockers: list[dict[str, str]] = []
        for blocker in blockers:
            if not isinstance(blocker, Mapping):
                continue
            code = _first_text(blocker.get("code"), "blocked") or "blocked"
            message = _first_text(blocker.get("message"), code) or code
            normalized_blockers.append({"code": code, "message": message})
        if normalized_blockers:
            normalized[file_id] = normalized_blockers
    return normalized


async def _hard_delete_generated_file(
    files_repo: Any,
    file_id: int,
    *,
    unregister_generated_file: Any | None = None,
) -> bool:
    if callable(unregister_generated_file):
        result = unregister_generated_file(file_id, hard_delete=True)
        deleted = await result if inspect.isawaitable(result) else result
        return bool(deleted)

    hard_delete_file = getattr(files_repo, "hard_delete_file", None)
    if not callable(hard_delete_file):
        raise ValueError("generated_files_repo_unavailable")
    result = hard_delete_file(file_id)
    deleted = await result if inspect.isawaitable(result) else result
    return bool(deleted)


def _matrix_variant_count_from_overrides(overrides: Mapping[str, Any]) -> int:
    raw_variant_count = overrides.get("variant_count", 1)
    return _strict_int_count(raw_variant_count, "invalid_matrix_variant_count")


def _strict_int_count(value: Any, error_code: str) -> int:
    if type(value) is not int:
        raise ValueError(error_code)
    return value


_PROMPT_BUDGET_FIELDS = {"character", "world_book", "pack", "slot", "total"}


def _prompt_budgets_from_request(value: Mapping[str, int] | None) -> PromptBudgets | None:
    if value is None:
        return None
    unknown_fields = set(value) - _PROMPT_BUDGET_FIELDS
    if unknown_fields:
        raise ValueError("invalid_prompt_budget")
    if any(type(budget) is not int for budget in value.values()):
        raise ValueError("invalid_prompt_budget")
    try:
        return PromptBudgets(**value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_prompt_budget") from exc


def _slot_create_spec(slot: VNAssetSlot) -> dict[str, Any]:
    return {
        "asset_type": slot.asset_type,
        "slot_key": slot.slot_key,
        "labels": slot.labels,
        "prompt_template": slot.prompt_template,
        "negative_prompt_template": slot.negative_prompt_template,
        "variant_count": slot.variant_count,
        "width": slot.width,
        "height": slot.height,
        "requires_review": slot.requires_review,
        "required_for_runtime": slot.required_for_runtime,
        "depends_on_slot_key": slot.depends_on_slot_key,
        "status": slot.status,
    }


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _join_prompt_parts(*values: Any) -> str | None:
    parts = [_first_text(value) for value in values]
    joined = "\n".join(part for part in parts if part)
    return joined or None


def _loads_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    return json.loads(value)


def _loads_optional_json(value: Any) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else None
