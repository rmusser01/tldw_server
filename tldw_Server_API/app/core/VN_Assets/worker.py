"""VN asset generation job handlers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from inspect import isawaitable
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.Image_Generation.adapter_registry import get_registry
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest
from tldw_Server_API.app.core.Storage.generated_file_helpers import save_and_register_vn_asset_image
from tldw_Server_API.app.core.VN_Assets.concurrency import get_default_backend_generation_gate
from tldw_Server_API.app.core.VN_Assets.constants import (
    SLOT_STATUS_FAILED,
    SLOT_STATUS_GENERATING,
    SLOT_STATUS_REVIEWING,
)
from tldw_Server_API.app.core.VN_Assets.jobs import (
    VN_ASSETS_DOMAIN,
    VN_ASSET_ENQUEUE_BATCH_JOB_TYPE,
    VN_ASSET_GENERATE_VARIANT_JOB_TYPE,
    create_generate_variant_job,
    vn_asset_batch_group,
)
from tldw_Server_API.app.core.VN_Assets.prompts import build_prompt_preview


class VNAssetGenerationWorker:
    """Synchronous handlers used by the async Jobs worker entrypoint."""

    def __init__(
        self,
        *,
        repo: VNAssetPacksRepository,
        jobs_manager: Any,
        image_registry: Any | None = None,
        backend_gate: Any | None = None,
        save_vn_asset_image: Any | None = None,
    ) -> None:
        self.repo = repo
        self.jobs_manager = jobs_manager
        self.image_registry = image_registry or get_registry()
        self.backend_gate = backend_gate or get_default_backend_generation_gate()
        self.save_vn_asset_image = save_vn_asset_image or save_and_register_vn_asset_image

    def handle_enqueue_batch(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        pack_id = _payload_int(payload, "pack_id")
        batch_id = _payload_int(payload, "batch_id")
        user_id = _payload_int(payload, "user_id")

        batch = self.repo.get_batch(batch_id)
        if batch is None or int(batch["pack_id"]) != pack_id:
            raise ValueError("vn_asset_batch_not_found")
        if int(batch["requested_by_user_id"]) != user_id:
            raise ValueError("vn_asset_job_owner_mismatch")

        slots = self.repo.list_slots(pack_id)
        options = _loads_json(batch.get("options_json"), {})
        slot_ids = {int(slot_id) for slot_id in options.get("slot_ids", [])}
        if slot_ids:
            slots = [slot for slot in slots if int(slot["id"]) in slot_ids]
        variant_count_override = options.get("variant_count")
        planned_count = sum(int(variant_count_override or slot["variant_count"]) for slot in slots)

        enqueued_count = 0
        try:
            for slot in slots:
                slot_variant_count = int(variant_count_override or slot["variant_count"])
                for variant_index in range(slot_variant_count):
                    create_generate_variant_job(
                        self.jobs_manager,
                        pack_id=pack_id,
                        slot_id=int(slot["id"]),
                        variant_index=variant_index,
                        batch_id=batch_id,
                        user_id=user_id,
                    )
                    enqueued_count += 1
            self.repo.update_batch(
                batch_id,
                {
                    "status": "enqueued",
                    "planned_count": planned_count,
                    "enqueued_count": enqueued_count,
                    "enqueue_error": None,
                    "total_slots": len(slots),
                    "total_variants": planned_count,
                },
            )
        except Exception as exc:
            self.repo.update_batch(
                batch_id,
                {
                    "status": "failed",
                    "planned_count": planned_count,
                    "enqueued_count": enqueued_count,
                    "enqueue_error": str(exc),
                },
            )
            raise

        logger.info(
            "VN asset batch fanout completed: batch_id={} pack_id={} slots={} variants={}",
            batch_id,
            pack_id,
            len(slots),
            enqueued_count,
        )
        return {
            "status": "enqueued",
            "batch_id": batch_id,
            "pack_id": pack_id,
            "enqueued_count": enqueued_count,
            "planned_count": planned_count,
        }

    async def handle_generate_variant(
        self,
        payload: Mapping[str, Any],
        *,
        job: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        pack_id = _payload_int(payload, "pack_id")
        slot_id = _payload_int(payload, "slot_id")
        variant_index = _payload_int(payload, "variant_index")
        batch_id = _payload_int(payload, "batch_id")
        user_id = _payload_int(payload, "user_id")

        batch = self.repo.get_batch(batch_id)
        if batch is None or int(batch["pack_id"]) != pack_id:
            raise ValueError("vn_asset_batch_not_found")
        if int(batch["requested_by_user_id"]) != user_id:
            raise ValueError("vn_asset_job_owner_mismatch")
        if _is_terminal_batch_status(batch["status"]):
            self._cancel_terminal_batch_jobs(
                user_id=user_id,
                pack_id=pack_id,
                batch_id=batch_id,
                current_job_id=_positive_int(_job_id(job)),
            )
            raise ValueError("vn_asset_batch_terminal")

        slot = self.repo.get_slot(slot_id)
        if slot is None or int(slot["pack_id"]) != pack_id:
            raise ValueError("slot_not_found")
        pack = self.repo.get_pack(pack_id)
        if pack is None or int(pack["owner_user_id"]) != user_id:
            raise ValueError("pack_not_found")
        character = self.repo.get_character(int(pack["primary_character_id"]))
        if character is None:
            raise ValueError("primary_character_not_found")

        try:
            return await self._generate_variant(
                pack=pack,
                slot=slot,
                batch=batch,
                character=character,
                variant_index=variant_index,
                user_id=user_id,
                job=job,
            )
        except Exception as exc:
            self._record_generation_failure(
                batch_id=batch_id,
                slot_id=slot_id,
                error=str(exc),
            )
            raise

    def handle_job(self, job: Mapping[str, Any]) -> dict[str, Any]:
        job_type = str(job.get("job_type") or "").strip()
        payload = job.get("payload") or {}
        if job_type == VN_ASSET_ENQUEUE_BATCH_JOB_TYPE:
            return self.handle_enqueue_batch(payload)
        if job_type == VN_ASSET_GENERATE_VARIANT_JOB_TYPE:
            raise ValueError("vn_asset_generate_variant_requires_async_handler")
        raise ValueError("unsupported_vn_asset_job_type")

    async def handle_job_async(self, job: Mapping[str, Any]) -> dict[str, Any]:
        job_type = str(job.get("job_type") or "").strip()
        payload = job.get("payload") or {}
        if job_type == VN_ASSET_ENQUEUE_BATCH_JOB_TYPE:
            return self.handle_enqueue_batch(payload)
        if job_type == VN_ASSET_GENERATE_VARIANT_JOB_TYPE:
            return await self.handle_generate_variant(payload, job=job)
        raise ValueError("unsupported_vn_asset_job_type")

    async def _generate_variant(
        self,
        *,
        pack: Mapping[str, Any],
        slot: Mapping[str, Any],
        batch: Mapping[str, Any],
        character: Mapping[str, Any],
        variant_index: int,
        user_id: int,
        job: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        pack_id = int(pack["id"])
        slot_id = int(slot["id"])
        batch_id = int(batch["id"])
        labels = _loads_json(slot.get("labels_json"), {})
        negative_prompt = _join_prompt_parts(
            pack.get("negative_prompt"),
            slot.get("negative_prompt_template"),
        )
        preview = build_prompt_preview(
            character=character,
            pack_style=pack.get("style_prompt"),
            pack_scenario=pack.get("scenario_notes"),
            negative_prompt=negative_prompt,
            style_lock=_loads_json(pack.get("style_lock_json"), {}),
            slot_template=slot.get("prompt_template"),
            labels=labels,
            world_book_entries=_world_book_entries_for_pack(self.repo, pack),
        )
        backend = self._resolve_backend(pack, slot)
        model = _first_text(slot.get("model_override"), pack.get("default_model"))
        width, height, image_format, extra_params = _generation_shape(pack, slot)
        request = ImageGenRequest(
            backend=backend,
            prompt=preview.prompt,
            negative_prompt=preview.negative_prompt or None,
            width=width,
            height=height,
            steps=_positive_int(extra_params.pop("steps", None)),
            cfg_scale=_float_or_none(extra_params.pop("cfg_scale", None)),
            seed=_variant_seed(slot, variant_index),
            sampler=_first_text(extra_params.pop("sampler", None)),
            model=model,
            format=image_format,
            extra_params=extra_params,
            request_id=f"vn_asset:{pack_id}:{slot_id}:{batch_id}:{variant_index}",
        )

        self.repo.update_slot(slot_id, {"status": SLOT_STATUS_GENERATING, "last_error": None})
        with self.backend_gate.try_acquire(backend, model=model) as lease:
            if not lease.acquired:
                raise ValueError("vn_asset_backend_busy")
            generation_result = self.image_registry.get_adapter(backend)
            if generation_result is None:
                raise ValueError("image_adapter_unavailable")
            image = await asyncio.to_thread(generation_result.generate, request)

        prompt_snapshot = {
            "prompt": preview.prompt,
            "negative_prompt": preview.negative_prompt,
            "token_estimates": preview.token_estimates,
            "omitted_source_counts": preview.omitted_source_counts,
            "warnings": list(preview.warnings),
        }
        context_snapshot = {
            "pack_id": pack_id,
            "slot_id": slot_id,
            "slot_key": slot.get("slot_key"),
            "batch_id": batch_id,
            "variant_index": variant_index,
            "primary_character_id": pack.get("primary_character_id"),
        }
        backend_metadata = {
            "backend": backend,
            "model": model,
            "request_id": request.request_id,
            "content_type": image.content_type,
            "bytes_len": image.bytes_len,
        }
        item = self.repo.create_item(
            pack_id=pack_id,
            slot_id=slot_id,
            variant_index=variant_index,
            mime_type=image.content_type,
            width=width,
            height=height,
            bytes=image.bytes_len,
            review_status="draft",
            source="generated",
            generation_job_id=_job_id(job),
            source_prompt_snapshot=prompt_snapshot,
            source_context_snapshot=context_snapshot,
            backend_metadata=backend_metadata,
        )
        item_id = int(item["id"])
        try:
            file_record = await _maybe_await(
                self.save_vn_asset_image(
                    user_id=user_id,
                    image_bytes=image.content,
                    image_format=_image_format_from_content_type(image.content_type, image_format),
                    pack_id=pack_id,
                    item_id=item_id,
                    asset_type=str(slot["asset_type"]),
                    labels=labels,
                )
            )
            item = self.repo.update_item_storage(
                item_id,
                generated_file_id=_positive_int(file_record.get("id")),
                storage_ref=_first_text(file_record.get("storage_path")),
                mime_type=_first_text(file_record.get("mime_type"), image.content_type),
                width=width,
                height=height,
                bytes=image.bytes_len,
                backend_metadata=backend_metadata,
            ) or item
        except Exception:
            self.repo.delete_item(item_id)
            raise
        self.repo.update_slot(slot_id, {"status": SLOT_STATUS_REVIEWING, "last_error": None})
        self._record_generation_success(batch_id=batch_id)
        logger.info(
            "VN asset variant generated: pack_id={} slot_id={} item_id={} backend={}",
            pack_id,
            slot_id,
            item["id"],
            backend,
        )
        return {
            "status": "draft_created",
            "pack_id": pack_id,
            "slot_id": slot_id,
            "item_id": int(item["id"]),
            "batch_id": batch_id,
            "generated_file_id": item["generated_file_id"],
        }

    def _resolve_backend(self, pack: Mapping[str, Any], slot: Mapping[str, Any]) -> str:
        requested_backend = _first_text(slot.get("backend_override"), pack.get("default_backend"))
        resolver = getattr(self.image_registry, "resolve_backend", None)
        backend = resolver(requested_backend) if callable(resolver) else requested_backend
        if not backend:
            raise ValueError("image_backend_unavailable")
        return str(backend)

    def _record_generation_success(self, *, batch_id: int) -> None:
        batch = self.repo.get_batch(batch_id)
        if batch is None:
            return
        completed_count = int(batch["completed_count"] or 0) + 1
        planned_count = int(batch["planned_count"] or batch["total_variants"] or 0)
        fields: dict[str, Any] = {"completed_count": completed_count}
        if not _is_terminal_batch_status(batch["status"]):
            if planned_count and completed_count >= planned_count:
                fields["status"] = "completed"
                fields["completed_at"] = _utc_now()
            else:
                fields["status"] = "processing"
        self.repo.update_batch(batch_id, fields)

    def _record_generation_failure(self, *, batch_id: int, slot_id: int, error: str) -> None:
        self.repo.update_slot(slot_id, {"status": SLOT_STATUS_FAILED, "last_error": error})
        batch = self.repo.get_batch(batch_id)
        if batch is None:
            return
        failed_count = int(batch["failed_count"] or 0) + 1
        self.repo.update_batch(batch_id, {"status": "failed", "failed_count": failed_count})

    def _cancel_terminal_batch_jobs(
        self,
        *,
        user_id: int,
        pack_id: int,
        batch_id: int,
        current_job_id: int | None,
    ) -> None:
        list_jobs = getattr(self.jobs_manager, "list_jobs", None)
        cancel_job = getattr(self.jobs_manager, "cancel_job", None)
        if not callable(list_jobs) or not callable(cancel_job):
            return
        batch_group = vn_asset_batch_group(user_id=user_id, pack_id=pack_id, batch_id=batch_id)
        for status in ("queued", "processing"):
            for job_row in list_jobs(
                domain=VN_ASSETS_DOMAIN,
                batch_group=batch_group,
                status=status,
                limit=500,
            ):
                job_id = _positive_int(job_row.get("id"))
                if job_id is None or job_id == current_job_id:
                    continue
                cancel_job(job_id, reason="vn_asset_batch_terminal")


def _payload_int(payload: Mapping[str, Any], key: str) -> int:
    try:
        return int(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing_{key}") from exc


_TERMINAL_BATCH_STATUSES = {"cancelled", "canceled", "completed", "failed"}


def _is_terminal_batch_status(status: Any) -> bool:
    return str(status or "").strip().lower() in _TERMINAL_BATCH_STATUSES


def _loads_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, Mapping):
        return dict(value)
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return default
    return loaded if isinstance(loaded, dict) else default


def _loads_json_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []


def _world_book_entries_for_pack(
    repo: VNAssetPacksRepository,
    pack: Mapping[str, Any],
) -> list[Any]:
    world_book_ids: list[int] = []
    for raw_id in _loads_json_list(pack.get("source_world_book_ids_json")):
        parsed_id = _positive_int(raw_id)
        if parsed_id is not None:
            world_book_ids.append(parsed_id)
    if not world_book_ids:
        return []

    try:
        from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService

        world_book_service = WorldBookService(repo.db)
        entries: list[Any] = []
        for world_book_id in world_book_ids:
            entries.extend(
                world_book_service.get_entries(
                    world_book_id=world_book_id,
                    enabled_only=True,
                )
            )
        return entries
    except Exception as exc:
        logger.warning(
            "Failed to load VN asset world-book context: pack_id={} error={}",
            pack.get("id"),
            exc,
        )
        return []


async def _maybe_await(value: Any) -> Any:
    if isawaitable(value):
        return await value
    return value


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


def _generation_shape(pack: Mapping[str, Any], slot: Mapping[str, Any]) -> tuple[int | None, int | None, str, dict[str, Any]]:
    dimensions = _loads_json(pack.get("default_dimensions_json"), {})
    width = _positive_int(slot.get("width")) or _positive_int(dimensions.get("width"))
    height = _positive_int(slot.get("height")) or _positive_int(dimensions.get("height"))
    image_format = _first_text(dimensions.get("format"), dimensions.get("image_format"), "png") or "png"
    extra_params = dimensions.get("extra_params")
    if not isinstance(extra_params, dict):
        extra_params = {}
    for key in ("steps", "cfg_scale", "sampler"):
        if key in dimensions and key not in extra_params:
            extra_params[key] = dimensions[key]
    return width, height, image_format.lower(), dict(extra_params)


def _variant_seed(slot: Mapping[str, Any], variant_index: int) -> int | None:
    seed_policy = _loads_json(slot.get("seed_policy_json"), {})
    seed = _positive_int(seed_policy.get("seed"))
    if seed is not None:
        return seed + variant_index
    base_seed = _positive_int(seed_policy.get("base_seed"))
    if base_seed is not None:
        return base_seed + variant_index
    return None


def _positive_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _image_format_from_content_type(content_type: str | None, default: str) -> str:
    normalized = str(content_type or "").lower()
    if normalized == "image/jpeg":
        return "jpg"
    if normalized.startswith("image/"):
        return normalized.split("/", 1)[1].split(";", 1)[0] or default
    return default


def _job_id(job: Mapping[str, Any] | None) -> str | None:
    if not job:
        return None
    return _first_text(job.get("id"), job.get("uuid"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
