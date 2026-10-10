"""VN asset generation job handlers."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from functools import partial
from hashlib import sha256
from inspect import isawaitable
from pathlib import Path
from threading import Event
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import SOURCE_FEATURE_VN_ASSETS
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.exceptions import LegacyDisplayReconciliationError, VNAssetGenerationError
from tldw_Server_API.app.core.Image_Generation.adapter_registry import get_registry
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest
from tldw_Server_API.app.core.Image_Generation.config import get_image_generation_config, resolve_image_generation_model
from tldw_Server_API.app.core.Image_Generation.exceptions import ImageGenerationError
from tldw_Server_API.app.core.Storage.file_integrity import generated_file_bytes_match
from tldw_Server_API.app.core.Storage.generated_file_helpers import save_and_register_vn_asset_image
from tldw_Server_API.app.core.VN_Assets.concurrency import get_default_backend_generation_gate
from tldw_Server_API.app.core.VN_Assets.constants import SLOT_STATUS_FAILED, SLOT_STATUS_REVIEWING
from tldw_Server_API.app.core.VN_Assets.jobs import (
    VN_ASSET_ENQUEUE_BATCH_JOB_TYPE,
    VN_ASSET_GENERATE_VARIANT_JOB_TYPE,
    VN_ASSETS_DOMAIN,
    VN_PACK_EXPORT_JOB_TYPE,
    VN_PACK_IMPORT_COMMIT_JOB_TYPE,
    VN_PACK_IMPORT_PREVIEW_JOB_TYPE,
    build_legacy_activity_reader,
    create_generate_variant_job,
    generate_variant_idempotency_key,
    legacy_delivery_fingerprint,
    vn_asset_batch_group,
    vn_asset_generation_jobs_queue,
)
from tldw_Server_API.app.core.VN_Assets.portability.exporter import VNPackExporter
from tldw_Server_API.app.core.VN_Assets.portability.importer import VNPackImporter
from tldw_Server_API.app.core.VN_Assets.portability.models import VNPackExportOptions
from tldw_Server_API.app.core.VN_Assets.portability.preview import VNPackImportPreviewer
from tldw_Server_API.app.core.VN_Assets.prompts import build_prompt_preview
from tldw_Server_API.app.core.VN_Assets.recipe import (
    RECIPE_VERSION,
    load_execution_recipe,
    load_recipe,
    slot_recipe,
)
from tldw_Server_API.app.core.VN_Assets.storage import (
    generated_file_matches_vn_asset,
    generated_file_size_bytes,
    resolve_vn_asset_storage_path,
    unlink_vn_asset_storage_file,
    vn_asset_source_ref,
)
from tldw_Server_API.app.services.storage_quota_service import get_storage_service


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
        generated_files_repo: Any | None = None,
        read_generated_file_bytes: Any | None = None,
        export_staging_root: Path | None = None,
        unregister_generated_file: Any | None = None,
        preflight_storage_quota: Any | None = None,
    ) -> None:
        self.repo = repo
        self.jobs_manager = jobs_manager
        self.repo.legacy_activity_reader = build_legacy_activity_reader(jobs_manager)
        self.image_registry = image_registry or get_registry()
        self.backend_gate = backend_gate or get_default_backend_generation_gate()
        self.save_vn_asset_image = save_vn_asset_image or save_and_register_vn_asset_image
        self.generated_files_repo = generated_files_repo
        self.read_generated_file_bytes = read_generated_file_bytes
        self.export_staging_root = export_staging_root
        self.unregister_generated_file = unregister_generated_file
        self.preflight_storage_quota = preflight_storage_quota

    def handle_enqueue_batch(
        self,
        payload: Mapping[str, Any],
        *,
        job: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fan out frozen variants without resetting outcomes or Jobs authority."""
        pack_id = _payload_int(payload, "pack_id")
        batch_id = _payload_int(payload, "batch_id")
        user_id = _payload_int(payload, "user_id")
        batch = self.repo.get_batch(batch_id)
        if batch is None or int(batch["pack_id"]) != pack_id:
            raise VNAssetGenerationError("vn_asset_batch_not_found", batch_id=batch_id, pack_id=pack_id)
        if int(batch["requested_by_user_id"]) != user_id:
            raise VNAssetGenerationError("vn_asset_job_owner_mismatch", batch_id=batch_id)
        recipe_version = int(batch.get("recipe_version") or 0)
        if recipe_version not in (0, 1):
            raise VNAssetGenerationError("vn_asset_recipe_version_unsupported", batch_id=batch_id)
        if _is_terminal_batch_status(batch["status"]) and not (
            recipe_version == 0 and _retryable_fanout_failure(batch)
        ):
            return {
                "status": str(batch["status"]), "batch_id": batch_id, "pack_id": pack_id,
                "enqueued_count": int(batch["enqueued_count"] or 0),
                "planned_count": int(batch["planned_count"] or 0),
            }

        authored_slots: list[dict[str, Any]] | None = None
        if batch.get("recipe_json") is not None:
            try:
                authored = load_recipe(batch["recipe_json"], pack_id=pack_id, owner_user_id=user_id)
                authored_slots = authored["slots"]
                if batch.get("execution_recipe_json") is None:
                    config = get_image_generation_config()
                    batch = self.repo.set_execution_recipe_if_absent(batch_id, {
                        "version": RECIPE_VERSION,
                        "slots": [self._resolve_slot_execution(slot, config) for slot in authored_slots],
                    })
                self._execution_recipe(batch)
            except Exception as exc:
                if recipe_version == 1:
                    self.repo.update_batch(batch_id, {"enqueue_error": str(exc)})
                else:
                    self.repo.fail_batch_fanout_if_active(
                        batch_id, error=str(exc),
                        failed_slot_ids=[
                            int(slot["slot_id"]) for slot in authored_slots or []
                            if int(slot["variant_count"]) > 0
                        ] if not _job_has_retry_remaining(job) else (),
                    )
                raise

        if recipe_version == 1:
            recipes = self.repo.list_batch_recipes(batch_id)
            planned_count = int(batch["planned_count"])
            if len(recipes) != planned_count:
                self.repo.fail_batch_integrity(batch_id, error="vn_asset_recipe_count_mismatch")
                raise VNAssetGenerationError("vn_asset_recipe_count_mismatch", batch_id=batch_id)
            variants = [(int(row["slot_id"]), int(row["variant_index"])) for row in recipes]
        elif authored_slots is not None:
            variants = [
                (int(slot["slot_id"]), index)
                for slot in authored_slots for index in range(int(slot["variant_count"]))
            ]
            planned_count = len(variants)
        else:
            slots = self.repo.list_slots(pack_id)
            options = _loads_json(batch.get("options_json"), {})
            slot_ids = {int(slot_id) for slot_id in options.get("slot_ids", [])}
            if slot_ids:
                slots = [slot for slot in slots if int(slot["id"]) in slot_ids]
            variant_count_override = options.get("variant_count")
            variants = [
                (int(slot["id"]), index)
                for slot in slots
                for index in range(int(variant_count_override or slot["variant_count"]))
            ]
            planned_count = len(variants)

        by_slot: dict[int, list[int]] = {}
        for slot_id, variant_index in variants:
            by_slot.setdefault(slot_id, []).append(variant_index)
        total_slots = len(by_slot)
        enqueued_count = 0
        fully_enqueued_slot_ids: set[int] = set()
        try:
            for slot_id, indexes in by_slot.items():
                for variant_index in indexes:
                    create_generate_variant_job(
                        self.jobs_manager, pack_id=pack_id, slot_id=slot_id,
                        variant_index=variant_index, batch_id=batch_id, user_id=user_id,
                    )
                    enqueued_count += 1
                fully_enqueued_slot_ids.add(slot_id)
            if recipe_version == 1:
                self.repo.mark_batch_enqueued(
                    batch_id, planned_count=planned_count,
                    enqueued_count=enqueued_count, total_slots=total_slots,
                )
                batch = self.repo.get_batch(batch_id) or batch
            else:
                batch = self.repo.complete_batch_fanout(
                    batch_id, planned_count=planned_count,
                    enqueued_count=enqueued_count, total_slots=total_slots,
                )
        except Exception as exc:
            if recipe_version == 1:
                self.repo.update_batch(batch_id, {
                    "planned_count": planned_count, "enqueued_count": enqueued_count,
                    "enqueue_error": str(exc),
                })
            else:
                self.repo.fail_batch_fanout_if_active(
                    batch_id, planned_count=planned_count, enqueued_count=enqueued_count,
                    error=str(exc),
                    failed_slot_ids=[
                        slot_id for slot_id in by_slot if slot_id not in fully_enqueued_slot_ids
                    ] if not _job_has_retry_remaining(job) else (),
                )
            raise

        logger.info(
            "VN asset batch fanout completed: batch_id={} pack_id={} slots={} variants={}",
            batch_id, pack_id, total_slots, enqueued_count,
        )
        return {
            "status": str(batch["status"]), "batch_id": batch_id, "pack_id": pack_id,
            "enqueued_count": enqueued_count, "planned_count": planned_count,
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
            raise VNAssetGenerationError("vn_asset_batch_not_found", batch_id=batch_id, pack_id=pack_id)
        if int(batch["requested_by_user_id"]) != user_id:
            raise VNAssetGenerationError("vn_asset_job_owner_mismatch", batch_id=batch_id)
        recipe_version = int(batch.get("recipe_version") or 0)
        lease_id = str(job.get("lease_id") or "") if job is not None else ""
        if recipe_version not in (0, 1):
            raise VNAssetGenerationError("vn_asset_recipe_version_unsupported", batch_id=batch_id)
        if recipe_version == 1 and batch["status"] == "cancelled":
            batch = await self.repo.run_worker_replay_operation(
                partial(self.repo.cancel_batch, batch_id)
            ) or batch
        outcome = await self.repo.get_variant_outcome_async(batch_id, slot_id, variant_index) if recipe_version == 1 else None
        if outcome is not None and outcome["outcome_status"] == "completed":
            replay = await self._replay_variant(
                batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                user_id=user_id, pack_id=pack_id,
            )
            if replay is not None:
                return replay
        if _is_terminal_batch_status(batch["status"]) and not (
            recipe_version == 0 and _retryable_fanout_failure(batch)
        ):
            if recipe_version == 1 and batch["status"] == "cancelled":
                # Clean sibling reservations before cancelling their only delivery path.
                recipes = await self.repo.run_worker_replay_operation(
                    partial(self.repo.list_batch_recipes, batch_id)
                )
                for recipe_row in recipes:
                    await self._cleanup_cancelled_variant_storage(
                        batch_id=batch_id, slot_id=int(recipe_row["slot_id"]),
                        variant_index=int(recipe_row["variant_index"]),
                        user_id=user_id, pack_id=pack_id,
                    )
            self._cancel_terminal_batch_jobs(
                user_id=user_id,
                pack_id=pack_id,
                batch_id=batch_id,
                current_job_id=_positive_int(_job_id(job)),
            )
            raise VNAssetGenerationError("vn_asset_batch_terminal", batch_id=batch_id, slot_id=slot_id)

        slot = self.repo.get_slot(slot_id)
        if slot is None or int(slot["pack_id"]) != pack_id:
            raise ValueError("slot_not_found")
        pack = self.repo.get_pack(pack_id)
        if pack is None or int(pack["owner_user_id"]) != user_id:
            raise ValueError("pack_not_found")
        recipe: Mapping[str, Any] | None = None
        character: Mapping[str, Any] | None = None
        if recipe_version == 1:
            recipe = await self.repo.run_worker_replay_operation(
                partial(self.repo.get_batch_recipe, batch_id, slot_id, variant_index)
            )
            if recipe is None:
                await self.repo.fail_batch_integrity_async(batch_id, error="vn_asset_recipe_not_found")
                raise VNAssetGenerationError(
                    "vn_asset_recipe_not_found", batch_id=batch_id,
                    slot_id=slot_id, variant_index=variant_index,
                )
        elif batch.get("recipe_json") is None:
            character = self.repo.get_character(int(pack["primary_character_id"]))
            if character is None:
                raise ValueError("primary_character_not_found")

        attempt_token: str | None = None
        claimed_item: Mapping[str, Any] | None = None
        if recipe_version == 1:
            if job is not None:
                await self.repo.run_worker_replay_operation(
                    partial(self._require_current_job_lease, job, user_id=user_id)
                )
            else:
                lease_id = "inline"
            if outcome is not None and outcome["outcome_status"] == "failed":
                raise VNAssetGenerationError("vn_asset_variant_failed", batch_id=batch_id, slot_id=slot_id)
            if recipe is None:
                raise VNAssetGenerationError("vn_asset_recipe_not_found", batch_id=batch_id, slot_id=slot_id)
            attempt_token = uuid.uuid4().hex
            try:
                claimed_item = await self.repo.run_worker_replay_operation(partial(
                    self.repo.claim_variant,
                    batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                    lease_id=lease_id, attempt_token=attempt_token,
                    allow_takeover=job is not None,
                    expected_claim_token=outcome.get("claim_token") if outcome else None,
                    validate_authority=(lambda: self._require_current_job_lease(job, user_id=user_id)) if job else None,
                    item_fields={
                        "pack_id": pack_id,
                        "generation_job_id": _job_id(job),
                        "source_prompt_snapshot": {
                            key: recipe[key] for key in (
                                "prompt", "negative_prompt", "token_estimates",
                                "omitted_source_counts", "warnings",
                            )
                        },
                        "source_context_snapshot": {
                            "pack_id": pack_id, "slot_id": slot_id,
                            "batch_id": batch_id, "variant_index": variant_index,
                            "slot_key": recipe["slot_key"],
                            "primary_character_id": recipe["primary_character_id"],
                        },
                    },
                ))
            except asyncio.CancelledError:
                if job is None:
                    await self.repo.run_worker_replay_operation(partial(
                        self.repo.release_variant_claim,
                        batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                        attempt_token=attempt_token,
                    ))
                raise

        reconciling = outcome is not None and outcome.get("item_id") is not None
        legacy_inline = recipe_version == 0 and job is None
        legacy_display_acquired = Event()
        legacy_status: str | None = None
        legacy_setup_complete = not legacy_inline
        try:
            if legacy_inline:
                await self.repo.run_worker_replay_operation(partial(
                    self.repo.begin_inline_legacy_display, batch_id, slot_id,
                    on_acquired=legacy_display_acquired.set,
                ))
                legacy_setup_complete = True
            if outcome is not None and outcome.get("item_id") is not None:
                if job is not None:
                    await self.repo.run_worker_replay_operation(
                        partial(self._require_current_job_lease, job, user_id=user_id)
                    )
                replay = await self._replay_variant(
                    batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                    user_id=user_id, pack_id=pack_id, attempt_token=attempt_token, job=job,
                )
                if replay is not None:
                    return replay
            reconciling = False
            result = await self._generate_variant(
                pack=pack,
                slot=slot,
                batch=batch,
                character=character,
                recipe=recipe,
                variant_index=variant_index,
                user_id=user_id,
                job=job,
                attempt_token=attempt_token,
                claimed_item=claimed_item,
            )
            legacy_status = SLOT_STATUS_REVIEWING
            return result
        except Exception as exc:
            if not legacy_setup_complete:
                raise
            legacy_status = SLOT_STATUS_FAILED
            if recipe_version == 1 and job is not None:
                await self.repo.run_worker_replay_operation(
                    partial(self._require_current_job_lease, job, user_id=user_id)
                )
            definitive_replay_failure = isinstance(exc, VNAssetGenerationError) and not exc.retryable
            definitive_generation_failure = (
                recipe_version == 1 and isinstance(exc, (ValueError, OSError))
                and not isinstance(exc, VNAssetGenerationError)
            )
            # Exhausted Jobs reconciliation fails the recipe without deleting stored bytes.
            if (definitive_generation_failure or definitive_replay_failure or not _job_has_retry_remaining(job)) and (
                not reconciling or definitive_replay_failure or job is not None
            ) and not (
                isinstance(exc, VNAssetGenerationError) and exc.retryable
                and (job is None or "max_retries" not in job)
            ):
                await self.repo.run_worker_replay_operation(partial(
                    self._record_generation_failure,
                    batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                    error=str(exc), attempt_token=attempt_token,
                    job=job, user_id=user_id,
                ))
            raise
        finally:
            if recipe_version == 0 and (not legacy_inline or legacy_display_acquired.is_set()):
                legacy_job_id = _positive_int(_job_id(job))
                try:
                    await self.repo.run_worker_replay_operation(partial(
                        self.repo.finish_legacy_display,
                        batch_id, slot_id, inline=legacy_inline, fallback_status=legacy_status,
                        finishing_delivery=(legacy_job_id, lease_id) if legacy_job_id is not None and lease_id else None,
                    ))
                except Exception as exc:  # noqa: BLE001 - display cannot alter generation's SDK disposition
                    # Frame metadata only: no messages, locals, source text or chained exceptions.
                    frames = []
                    error_type = type(exc)
                    trace = exc.__traceback__
                    if isinstance(exc, LegacyDisplayReconciliationError):
                        error_type = exc.error_type
                        trace = exc.error_traceback
                    while trace is not None:
                        frames.append({
                            "file": trace.tb_frame.f_code.co_filename,
                            "function": trace.tb_frame.f_code.co_name,
                            "line": trace.tb_lineno,
                        })
                        trace = trace.tb_next
                    logger.bind(
                        operation="finish_legacy_display", user_id=user_id, pack_id=pack_id,
                        batch_id=batch_id, slot_id=slot_id, job_id=legacy_job_id,
                        error_type=error_type.__name__, traceback_frames=frames,
                    ).warning("VN legacy display reconciliation failed")
            if attempt_token is not None and job is None:
                await self.repo.run_worker_replay_operation(partial(
                    self.repo.release_variant_claim,
                    batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                    attempt_token=attempt_token,
                ))

    async def handle_failed_job(self, job: Mapping[str, Any], error: Exception) -> None:
        """Reconcile the exact V1 claim after Jobs durably terminates its delivery."""
        if job.get("job_type") != VN_ASSET_GENERATE_VARIANT_JOB_TYPE:
            return
        payload = job.get("payload") or {}
        user_id = _payload_int(payload, "user_id")
        pack_id = _payload_int(payload, "pack_id")
        batch_id = _payload_int(payload, "batch_id")
        slot_id = _payload_int(payload, "slot_id")
        variant_index = _payload_int(payload, "variant_index")
        lease_id = str(job.get("lease_id") or "")
        job_id = _positive_int(_job_id(job))
        expected_scope = {
            "owner_user_id": str(user_id),
            "domain": VN_ASSETS_DOMAIN,
            "queue": vn_asset_generation_jobs_queue(),
            "job_type": VN_ASSET_GENERATE_VARIANT_JOB_TYPE,
            "batch_group": vn_asset_batch_group(user_id=user_id, pack_id=pack_id, batch_id=batch_id),
            "idempotency_key": generate_variant_idempotency_key(
                user_id=user_id, pack_id=pack_id, batch_id=batch_id,
                slot_id=slot_id, variant_index=variant_index,
            ),
        }
        if (
            not lease_id or job_id is None or not job.get("uuid")
            or any(job.get(key) != value for key, value in expected_scope.items())
        ):
            return

        def reconcile() -> None:
            outcome = self.repo.get_variant_outcome(batch_id, slot_id, variant_index)
            if (
                not outcome or outcome["outcome_status"] != "planned"
                or outcome.get("claim_lease_id") != lease_id or not outcome.get("claim_token")
            ):
                return
            attempt_token = str(outcome["claim_token"])

            def is_current_terminal_claim() -> bool:
                current = self.jobs_manager.get_job_or_archived_by_uuid(
                    str(job["uuid"]), domain=VN_ASSETS_DOMAIN, owner_user_id=str(user_id),
                )
                if (
                    not current or current.get("uuid") != job["uuid"]
                    or current.get("status") not in {"failed", "quarantined"}
                    or current.get("completion_token") != lease_id
                    or (not current.get("archived") and current.get("id") != job_id)
                    or current.get("payload") != payload
                    or any(current.get(key) != value for key, value in expected_scope.items())
                ):
                    return False
                batch = self.repo.get_batch(batch_id)
                pack = self.repo.get_pack(pack_id)
                claimed = self.repo.get_variant_outcome(batch_id, slot_id, variant_index)
                item = self.repo.get_item(int(claimed["item_id"])) if claimed and claimed.get("item_id") else None
                return bool(
                    batch and int(batch["recipe_version"] or 0) == 1
                    and int(batch["pack_id"]) == pack_id and int(batch["requested_by_user_id"]) == user_id
                    and pack and int(pack["owner_user_id"]) == user_id
                    and claimed and claimed["outcome_status"] == "planned"
                    and claimed.get("claim_token") == attempt_token and claimed.get("claim_lease_id") == lease_id
                    and item and int(item["pack_id"]) == pack_id and int(item["slot_id"]) == slot_id
                    and _positive_int(item.get("generation_job_id")) == job_id
                )

            if not is_current_terminal_claim():
                return

            def validate_authority() -> None:
                if not is_current_terminal_claim():
                    raise VNAssetGenerationError("vn_asset_job_lease_lost", retryable=True, job_id=job_id)

            # Jobs cleared the live lease; its completion token and the V1 claim bind this failure.
            self.repo.fail_variant(
                batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                error=str(error), attempt_token=attempt_token, validate_authority=validate_authority,
            )

        await self.repo.run_worker_replay_operation(reconcile)

    def _require_current_job_lease(self, job: Mapping[str, Any], *, user_id: int) -> None:
        """Validate job's live lease for user_id, returning None on admission.

        Raises retryable VNAssetGenerationError for missing, cancelled,
        cancellation-requested, replaced, malformed, or expired leases; Jobs
        read failures propagate. Admission is
        a snapshot, so each VN mutation calls this again under its write lock.
        """
        lease_id = str(job.get("lease_id") or "")
        job_id = _positive_int(_job_id(job))
        if not lease_id or job_id is None:
            raise VNAssetGenerationError("vn_asset_job_lease_lost", retryable=True, job_id=job_id)
        current = self.jobs_manager.get_job(job_id, owner_user_id=str(user_id))
        if (
            current is None or current.get("status") != "processing"
            or current.get("cancel_requested_at") is not None
            or str(current.get("lease_id") or "") != lease_id
        ):
            raise VNAssetGenerationError("vn_asset_job_lease_lost", retryable=True, job_id=job_id)
        try:
            leased_until = datetime.fromisoformat(str(current.get("leased_until") or "").replace("Z", "+00:00"))
            if leased_until.tzinfo is None:
                leased_until = leased_until.replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise VNAssetGenerationError("vn_asset_job_lease_lost", retryable=True, job_id=job_id) from exc
        if leased_until <= datetime.now(timezone.utc):
            raise VNAssetGenerationError("vn_asset_job_lease_lost", retryable=True, job_id=job_id)

    async def _replay_variant(
        self,
        *,
        batch_id: int,
        slot_id: int,
        variant_index: int,
        user_id: int,
        pack_id: int,
        allow_publication: bool = True,
        attempt_token: str | None = None,
        job: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Return the variant result for the supplied IDs, or None without stored bytes.

        allow_publication=False limits this to completed replay. attempt_token
        and job fence reconciliation/publication. Raises VNAssetGenerationError
        for invalid persisted state or lost authority; storage/DB errors propagate.
        """
        outcome = await self.repo.get_variant_outcome_async(batch_id, slot_id, variant_index)
        if outcome is None:
            await self.repo.fail_batch_integrity_async(batch_id, error="vn_asset_recipe_not_found")
            raise VNAssetGenerationError("vn_asset_recipe_not_found", batch_id=batch_id, slot_id=slot_id)
        if outcome["outcome_status"] == "failed":
            if allow_publication:
                raise VNAssetGenerationError("vn_asset_variant_failed", batch_id=batch_id, slot_id=slot_id)
            return None
        if outcome["outcome_status"] != "completed" and not allow_publication:
            return None
        if outcome.get("deleted_item_json") is not None:
            return _deleted_variant_result(
                outcome, batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                user_id=user_id, pack_id=pack_id,
            )
        item_id = _positive_int(outcome.get("item_id"))
        if item_id is None:
            return None
        item = await self.repo.run_worker_replay_operation(partial(self.repo.get_item, item_id))
        if item is None or int(item["pack_id"]) != pack_id or int(item["slot_id"]) != slot_id:
            raise VNAssetGenerationError("vn_asset_recipe_item_missing", batch_id=batch_id, item_id=item_id)
        files_repo = self.generated_files_repo
        if files_repo is None:
            storage_service = await get_storage_service()
            files_repo = await storage_service.get_generated_files_repo()
        attached_file_id = item.get("generated_file_id")
        if attached_file_id is None:
            if outcome["outcome_status"] == "completed":
                raise VNAssetGenerationError(
                    "vn_asset_item_storage_missing", batch_id=batch_id, item_id=item_id,
                )
            file_record = await files_repo.get_file_by_source_ref(
                user_id=user_id,
                source_feature=SOURCE_FEATURE_VN_ASSETS,
                source_ref=vn_asset_source_ref(item_id),
            )
            if file_record is None:
                return None
        else:
            file_record = await files_repo.get_file_by_id(int(attached_file_id))
        try:
            valid_record = (
                file_record is not None
                and generated_file_matches_vn_asset(file_record, user_id=user_id, item_id=item_id)
                and _positive_int(file_record.get("id")) is not None
                and (attached_file_id is None or (
                    int(file_record["id"]) == int(attached_file_id)
                    and file_record.get("storage_path") == item.get("storage_ref")
                    and generated_file_size_bytes(file_record) == int(item.get("bytes") or 0)
                ))
            )
            valid_bytes = valid_record and await asyncio.to_thread(
                _replay_file_bytes_present, file_record, user_id=user_id,
            )
        except OSError as exc:
            raise VNAssetGenerationError(
                "vn_asset_item_storage_missing", retryable=True,
                batch_id=batch_id, slot_id=slot_id, variant_index=variant_index, item_id=item_id,
                operation="replay_variant",
            ) from exc
        except (TypeError, ValueError):
            valid_bytes = False
        if not valid_bytes:
            raise VNAssetGenerationError(
                "vn_asset_item_storage_missing",
                batch_id=batch_id, slot_id=slot_id, variant_index=variant_index, item_id=item_id,
                operation="replay_variant",
            )
        if outcome["outcome_status"] == "completed":
            return _generated_variant_result(item, batch_id=batch_id)

        def reconcile() -> dict[str, Any]:
            """Finish synchronous reconciliation/fences using only thread-owned handles."""
            reconciled = item
            if attached_file_id is None:
                recipe = self.repo.get_batch_recipe(batch_id, slot_id, variant_index)
                if recipe is None:
                    raise VNAssetGenerationError("vn_asset_recipe_not_found", batch_id=batch_id, slot_id=slot_id)
                if job is not None:
                    self._require_current_job_lease(job, user_id=user_id)
                reconciled = self.repo.update_item_storage(
                    item_id,
                    generated_file_id=int(file_record["id"]),
                    storage_ref=str(file_record["storage_path"]),
                    mime_type=str(file_record.get("mime_type") or "image/png"),
                    width=_positive_int(recipe.get("width")),
                    height=_positive_int(recipe.get("height")),
                    bytes=generated_file_size_bytes(file_record),
                    batch_id=batch_id if attempt_token else None,
                    slot_id=slot_id if attempt_token else None,
                    variant_index=variant_index if attempt_token else None,
                    attempt_token=attempt_token,
                    validate_authority=(lambda: self._require_current_job_lease(job, user_id=user_id)) if job else None,
                ) or reconciled
            if job is not None:
                self._require_current_job_lease(job, user_id=user_id)
            return self.repo.complete_variant(
                batch_id=batch_id, slot_id=slot_id,
                variant_index=variant_index, item_id=item_id,
                attempt_token=attempt_token,
                validate_authority=(lambda: self._require_current_job_lease(job, user_id=user_id)) if job else None,
            )

        item = await self.repo.run_worker_replay_operation(reconcile)
        return _generated_variant_result(item, batch_id=batch_id)

    def handle_job(self, job: Mapping[str, Any]) -> dict[str, Any]:
        job_type = str(job.get("job_type") or "").strip()
        payload = job.get("payload") or {}
        if job_type == VN_ASSET_ENQUEUE_BATCH_JOB_TYPE:
            return self.handle_enqueue_batch(payload, job=job)
        if job_type == VN_ASSET_GENERATE_VARIANT_JOB_TYPE:
            raise VNAssetGenerationError("vn_asset_generate_variant_requires_async_handler")
        if job_type == VN_PACK_EXPORT_JOB_TYPE:
            raise ValueError("vn_pack_export_requires_async_handler")
        if job_type == VN_PACK_IMPORT_PREVIEW_JOB_TYPE:
            raise ValueError("vn_pack_import_preview_requires_async_handler")
        if job_type == VN_PACK_IMPORT_COMMIT_JOB_TYPE:
            raise ValueError("vn_pack_import_commit_requires_async_handler")
        raise ValueError("unsupported_vn_asset_job_type")

    async def handle_job_async(self, job: Mapping[str, Any]) -> dict[str, Any]:
        job_type = str(job.get("job_type") or "").strip()
        payload = job.get("payload") or {}
        if job_type == VN_ASSET_ENQUEUE_BATCH_JOB_TYPE:
            return self.handle_enqueue_batch(payload, job=job)
        if job_type == VN_ASSET_GENERATE_VARIANT_JOB_TYPE:
            return await self.handle_generate_variant(payload, job=job)
        if job_type == VN_PACK_EXPORT_JOB_TYPE:
            return await self.handle_export_pack(payload, job=job)
        if job_type == VN_PACK_IMPORT_PREVIEW_JOB_TYPE:
            return await self.handle_import_preview(payload, job=job)
        if job_type == VN_PACK_IMPORT_COMMIT_JOB_TYPE:
            return await self.handle_import_commit(payload, job=job)
        raise ValueError("unsupported_vn_asset_job_type")

    async def handle_export_pack(
        self,
        payload: Mapping[str, Any],
        *,
        job: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        pack_id = _payload_int(payload, "pack_id")
        user_id = _payload_int(payload, "user_id")

        pack = self.repo.get_pack(pack_id)
        if pack is None or int(pack["owner_user_id"]) != user_id:
            raise ValueError("pack_not_found")
        portability_job_id = _payload_int(payload, "portability_job_id", default=0)
        portability_job = (
            self.repo.get_portability_job(portability_job_id, owner_user_id=user_id)
            if portability_job_id > 0
            else self.repo.get_portability_job_by_job_id(str((job or {}).get("id")), owner_user_id=user_id)
        )
        if portability_job is None or int(portability_job["pack_id"] or 0) != pack_id:
            raise ValueError("vn_pack_portability_job_not_found")
        portability_job_id = int(portability_job["id"])
        if self.generated_files_repo is None or self.read_generated_file_bytes is None:
            raise ValueError("vn_pack_export_storage_unavailable")

        job_id = str(portability_job["job_id"])
        self.repo.update_portability_job(
            job_id,
            {"status": "processing", "stage": "collecting_metadata", "progress": {"pack_id": pack_id}},
            owner_user_id=user_id,
        )

        def _progress(stage: str, progress: dict[str, Any]) -> None:
            self.repo.update_portability_job(
                job_id,
                {"status": "processing", "stage": stage, "progress": progress},
                owner_user_id=user_id,
            )

        try:
            exporter = VNPackExporter(
                repo=self.repo,
                owner_user_id=user_id,
                generated_files_repo=self.generated_files_repo,
                read_generated_file_bytes=self.read_generated_file_bytes,
                staging_root=self._export_staging_root(user_id),
            )
            result = await exporter.export_pack(
                pack_id=pack_id,
                options=_export_options(payload.get("options")),
                progress=_progress,
            )
        except Exception as exc:
            self.repo.update_portability_job(
                job_id,
                {
                    "status": "failed",
                    "stage": "failed",
                    "error_code": "export_failed",
                    "error_message": str(exc),
                },
                owner_user_id=user_id,
            )
            raise

        expires_at = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        self.repo.update_portability_job(
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "archive_path": str(result.archive_path),
                "archive_sha256": result.archive_sha256,
                "canonical_payload_fingerprint": result.canonical_payload_fingerprint,
                "warnings": result.warnings,
                "progress": {"file_size_bytes": result.file_size_bytes},
                "expires_at": expires_at,
            },
            owner_user_id=user_id,
        )
        return {
            "status": "exported",
            "pack_id": pack_id,
            "portability_job_id": portability_job_id,
            "archive_path": str(result.archive_path),
            "archive_sha256": result.archive_sha256,
            "canonical_payload_fingerprint": result.canonical_payload_fingerprint,
            "file_size_bytes": result.file_size_bytes,
            "warnings": result.warnings,
        }

    async def handle_import_preview(
        self,
        payload: Mapping[str, Any],
        *,
        job: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        preview_id = _payload_int(payload, "preview_id")
        user_id = _payload_int(payload, "user_id")
        preview = self.repo.get_import_preview(preview_id, owner_user_id=user_id)
        if preview is None:
            raise ValueError("vn_pack_import_preview_not_found")
        job_id = str(preview["job_id"])
        portability_job = self.repo.get_portability_job_by_job_id(job_id, owner_user_id=user_id)
        preview_status = str(preview["status"])
        if preview_status in {"deleted", "cancelled"}:
            if portability_job is not None and (
                str(portability_job["status"]) != "cancelled"
                or str(portability_job["stage"]) != preview_status
            ):
                self.repo.update_portability_job(
                    job_id,
                    {"status": "cancelled", "stage": preview_status},
                    owner_user_id=user_id,
                )
            return {
                "status": "cancelled",
                "preview_id": preview_id,
                "archive_path": str(preview.get("archive_path") or ""),
            }
        archive_path = Path(str(payload.get("archive_path") or preview.get("archive_path") or ""))
        if not archive_path.is_file():
            raise ValueError("vn_pack_import_archive_not_found")

        self.repo.update_import_preview(
            preview_id,
            {"status": "processing", "archive_path": str(archive_path)},
            owner_user_id=user_id,
        )
        if portability_job is not None:
            self.repo.update_portability_job(
                job_id,
                {"status": "processing", "stage": "validating_archive"},
                owner_user_id=user_id,
            )

        def _progress(stage: str, progress: dict[str, Any]) -> None:
            self.repo.update_portability_job(
                job_id,
                {"status": "processing", "stage": stage, "progress": progress},
                owner_user_id=user_id,
            )

        try:
            previewer = VNPackImportPreviewer(repo=self.repo)
            result = await previewer.create_preview(
                archive_path=archive_path,
                owner_user_id=user_id,
                progress=_progress,
            )
        except Exception as exc:
            self.repo.update_import_preview(
                preview_id,
                {"status": "failed"},
                owner_user_id=user_id,
            )
            self.repo.update_portability_job(
                job_id,
                {
                    "status": "failed",
                    "stage": "failed",
                    "error_code": "import_preview_failed",
                    "error_message": str(exc),
                },
                owner_user_id=user_id,
            )
            raise

        expires_at = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        self.repo.update_import_preview(
            preview_id,
            {
                "status": "completed",
                "archive_sha256": result["archive_sha256"],
                "canonical_payload_fingerprint": result["canonical_payload_fingerprint"],
                "schema_version": result["schema_version"],
                "bundle_summary": result["bundle_summary"],
                "validation_warnings": result["validation_warnings"],
                "conflicts": result["conflicts"],
                "proposed_plan": result["proposed_plan"],
                "quota_estimate": result["quota_estimate"],
                "required_choices": result["required_choices"],
                "expires_at": expires_at,
            },
            owner_user_id=user_id,
        )
        self.repo.update_portability_job(
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "archive_path": str(archive_path),
                "archive_sha256": result["archive_sha256"],
                "canonical_payload_fingerprint": result["canonical_payload_fingerprint"],
                "warnings": result["validation_warnings"],
                "progress": result["bundle_summary"],
                "expires_at": expires_at,
            },
            owner_user_id=user_id,
        )
        return {
            **result,
            "status": "previewed",
            "preview_id": preview_id,
            "archive_path": str(archive_path),
        }

    async def handle_import_commit(
        self,
        payload: Mapping[str, Any],
        *,
        job: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        import_id = _payload_int(payload, "import_id")
        preview_id = _payload_int(payload, "preview_id")
        user_id = _payload_int(payload, "user_id")
        trust_mode = _payload_text(payload, "trust_mode")
        target_mode = _payload_text(payload, "target_mode")
        character_action = _payload_text(payload, "character_action")
        target_character_id = _payload_optional_int(payload, "target_character_id")
        target_pack_id = _payload_optional_int(payload, "target_pack_id")
        conflict_decisions = payload.get("conflict_decisions")
        if not isinstance(conflict_decisions, Mapping):
            conflict_decisions = {}

        journal = self.repo.get_import_journal(import_id, owner_user_id=user_id)
        if journal is None or int(journal["preview_id"]) != preview_id:
            raise ValueError("vn_pack_import_journal_not_found")
        preview = self.repo.get_import_preview(preview_id, owner_user_id=user_id)
        if preview is None:
            raise ValueError("vn_pack_import_preview_not_found")
        job_id = str((job or {}).get("id") or journal["job_id"])
        portability_job = self.repo.get_portability_job_by_job_id(job_id, owner_user_id=user_id)
        if portability_job is None or portability_job.get("operation") != "import_commit":
            raise ValueError("vn_pack_import_commit_job_not_found")

        self.repo.update_import_journal(
            import_id,
            {"status": "processing", "stage": "revalidating_preview", "job_id": job_id},
            owner_user_id=user_id,
        )
        self.repo.update_portability_job(
            job_id,
            {
                "status": "processing",
                "stage": "revalidating_preview",
                "progress": {"preview_id": preview_id, "import_id": import_id},
            },
            owner_user_id=user_id,
        )

        def _progress(stage: str, progress: dict[str, Any]) -> None:
            self.repo.update_portability_job(
                job_id,
                {"status": "processing", "stage": stage, "progress": progress},
                owner_user_id=user_id,
            )

        try:
            importer = VNPackImporter(
                repo=self.repo,
                owner_user_id=user_id,
                save_vn_asset_image=self.save_vn_asset_image,
                unregister_generated_file=self.unregister_generated_file,
                preflight_storage_quota=self.preflight_storage_quota,
            )
            result = await importer.import_pack(
                preview_id=preview_id,
                job_id=job_id,
                trust_mode=trust_mode,
                target_mode=target_mode,
                character_action=character_action,
                target_character_id=target_character_id,
                target_pack_id=target_pack_id,
                conflict_decisions=conflict_decisions,
                journal_id=import_id,
                progress=_progress,
            )
        except Exception as exc:
            self.repo.update_import_journal(
                import_id,
                {
                    "status": "failed",
                    "stage": "failed",
                    "error_code": "import_failed",
                    "error_message": str(exc),
                },
                owner_user_id=user_id,
            )
            self.repo.update_portability_job(
                job_id,
                {
                    "status": "failed",
                    "stage": "failed",
                    "error_code": "import_failed",
                    "error_message": str(exc),
                },
                owner_user_id=user_id,
            )
            raise

        self.repo.update_import_journal(
            import_id,
            {"target_pack_id": int(result["pack_id"])},
            owner_user_id=user_id,
        )
        self.repo.update_portability_job(
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "pack_id": int(result["pack_id"]),
                "progress": {
                    "pack_id": int(result["pack_id"]),
                    "created_records": result.get("created_records", {}),
                },
            },
            owner_user_id=user_id,
        )
        return result

    def _export_staging_root(self, user_id: int) -> Path:
        if self.export_staging_root is not None:
            return Path(self.export_staging_root)
        raise ValueError("vn_pack_export_staging_root_unavailable")

    async def _generate_variant(
        self,
        *,
        pack: Mapping[str, Any],
        slot: Mapping[str, Any],
        batch: Mapping[str, Any],
        character: Mapping[str, Any] | None,
        recipe: Mapping[str, Any] | None,
        variant_index: int,
        user_id: int,
        job: Mapping[str, Any] | None,
        attempt_token: str | None = None,
        claimed_item: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        pack_id = int(pack["id"])
        slot_id = int(slot["id"])
        batch_id = int(batch["id"])
        if batch.get("recipe_json") is not None:
            authored = load_recipe(batch["recipe_json"], pack_id=pack_id, owner_user_id=user_id)
            authored_slot = slot_recipe(authored, slot_id)
            if variant_index < 0 or variant_index >= int(authored_slot["variant_count"]):
                raise ValueError("vn_asset_recipe_variant_mismatch")
            if recipe is None:
                recipe = authored_variant_recipe(authored, authored_slot, variant_index)
            if batch.get("execution_recipe_json") is None:
                config = get_image_generation_config()
                execution_recipe = {
                    "version": RECIPE_VERSION,
                    "slots": [self._resolve_slot_execution(entry, config) for entry in authored["slots"]],
                }

                def pin_execution_recipe() -> dict[str, Any]:
                    """Admit the current claim and pin settings in one owning transaction."""
                    with self.repo.db.transaction():
                        if attempt_token is not None:
                            self.repo.start_variant_generation(
                                batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                                attempt_token=attempt_token,
                                validate_authority=(
                                    lambda: self._require_current_job_lease(job, user_id=user_id)
                                ) if job else None,
                            )
                        return self.repo.set_execution_recipe_if_absent(batch_id, execution_recipe)

                batch = await self.repo.run_worker_replay_operation(pin_execution_recipe)
            execution = slot_recipe(self._execution_recipe(batch), slot_id)
            backend = str(execution["backend"])
            model = execution.get("model")
            metadata_model = model
            if backend == "stable_diffusion_cpp":
                mode, configured_path = _local_model_state(get_image_generation_config())
                if mode != execution.get("local_model_mode"):
                    raise ValueError("vn_asset_local_model_changed")
                if model is None:
                    if _path_digest(configured_path) != execution.get("local_model_path_sha256"):
                        raise ValueError("vn_asset_local_model_changed")
                    model = configured_path
        else:
            if recipe is None:
                if character is None:
                    raise ValueError("primary_character_not_found")
                recipe = {
                    **build_slot_recipe(self.repo, pack, slot, character),
                    "seed": variant_seed(slot, variant_index),
                }
            backend = self._resolve_backend(recipe.get("requested_backend"))
            model = _first_text(recipe.get("model"))
            metadata_model = model
        labels = dict(recipe["labels"])
        width = _positive_int(recipe.get("width"))
        height = _positive_int(recipe.get("height"))
        image_format = str(recipe["format"])
        request = ImageGenRequest(
            backend=backend,
            prompt=str(recipe["prompt"]),
            negative_prompt=_first_text(recipe.get("negative_prompt")),
            width=width,
            height=height,
            steps=_positive_int(recipe.get("steps")),
            cfg_scale=_float_or_none(recipe.get("cfg_scale")),
            seed=recipe.get("seed"),
            sampler=_first_text(recipe.get("sampler")),
            model=model,
            format=image_format,
            extra_params=dict(recipe.get("extra_params") or {}),
            request_id=f"vn_asset:{pack_id}:{slot_id}:{batch_id}:{variant_index}",
        )

        if attempt_token is not None:
            await self.repo.run_worker_replay_operation(partial(
                self.repo.start_variant_generation,
                batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                attempt_token=attempt_token,
                validate_authority=(lambda: self._require_current_job_lease(job, user_id=user_id)) if job else None,
            ))
        else:
            self.repo.mark_slot_generation_started(slot_id, batch_id)
        with self.backend_gate.try_acquire(backend, model=model) as lease:
            if not lease.acquired:
                raise VNAssetGenerationError(
                    "vn_asset_backend_busy", retryable=attempt_token is not None,
                    batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                )
            generation_result = self.image_registry.get_adapter(backend)
            if generation_result is None:
                raise VNAssetGenerationError("image_adapter_unavailable", batch_id=batch_id, slot_id=slot_id)
            if job is not None and attempt_token is not None:
                await self.repo.run_worker_replay_operation(
                    partial(self._require_current_job_lease, job, user_id=user_id)
                )
            try:
                image = await asyncio.to_thread(generation_result.generate, request)
            except OSError as exc:
                # Adapter I/O failures use the retry contract, unlike repository failures.
                raise ImageGenerationError("image adapter generation failed") from exc

        if attempt_token is not None:
            if job is not None:
                await self.repo.run_worker_replay_operation(
                    partial(self._require_current_job_lease, job, user_id=user_id)
                )
            outcome = await self.repo.get_variant_outcome_async(batch_id, slot_id, variant_index)
            current_batch = await self.repo.run_worker_replay_operation(partial(self.repo.get_batch, batch_id))
            if current_batch is None or _is_terminal_batch_status(current_batch["status"]):
                raise VNAssetGenerationError("vn_asset_batch_terminal", retryable=True, batch_id=batch_id)
            if outcome is None or outcome.get("claim_token") != attempt_token:
                raise VNAssetGenerationError(
                    "vn_asset_variant_claim_lost", retryable=True,
                    batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                )

        prompt_snapshot = {
            "prompt": recipe["prompt"],
            "negative_prompt": recipe["negative_prompt"],
            "token_estimates": recipe["token_estimates"],
            "omitted_source_counts": recipe["omitted_source_counts"],
            "warnings": recipe["warnings"],
        }
        context_snapshot = {
            "pack_id": pack_id,
            "slot_id": slot_id,
            "slot_key": recipe["slot_key"],
            "batch_id": batch_id,
            "variant_index": variant_index,
            "primary_character_id": recipe["primary_character_id"],
        }
        if int(batch.get("recipe_version") or 0) == 0:
            fingerprint = legacy_delivery_fingerprint(job)
            if fingerprint is not None:
                context_snapshot["legacy_delivery_fingerprint"] = fingerprint
        backend_metadata = {
            "backend": backend,
            "model": metadata_model,
            "request_id": request.request_id,
            "content_type": image.content_type,
            "bytes_len": image.bytes_len,
        }
        item_fields = {
            "pack_id": pack_id,
            "mime_type": image.content_type,
            "width": width,
            "height": height,
            "bytes": image.bytes_len,
            "source": "generated",
            "generation_job_id": _job_id(job),
            "source_prompt_snapshot": prompt_snapshot,
            "source_context_snapshot": context_snapshot,
            "backend_metadata": backend_metadata,
        }
        if int(batch.get("recipe_version") or 0) == 1:
            item = claimed_item or self.repo.reserve_variant_item(
                batch_id=batch_id, slot_id=slot_id,
                variant_index=variant_index, item_fields=item_fields,
            )
        else:
            item = self.repo.create_item(
                slot_id=slot_id,
                variant_index=variant_index,
                review_status="draft",
                **item_fields,
            )
        item_id = int(item["id"])
        file_record: dict[str, Any] | None = None
        try:
            file_record = await _maybe_await(
                self.save_vn_asset_image(
                    user_id=user_id,
                    image_bytes=image.content,
                    image_format=_image_format_from_content_type(image.content_type, image_format),
                    pack_id=pack_id,
                    item_id=item_id,
                    asset_type=str(recipe["asset_type"]),
                    labels=labels,
                )
            )
            if attempt_token is not None and job is not None:
                await self.repo.run_worker_replay_operation(
                    partial(self._require_current_job_lease, job, user_id=user_id)
                )
            stored_bytes = generated_file_size_bytes(file_record, fallback=image.bytes_len)
            backend_metadata = {
                **backend_metadata,
                "content_type": _first_text(file_record.get("mime_type"), image.content_type),
                "bytes_len": stored_bytes,
            }
            item = self.repo.update_item_storage(
                item_id,
                generated_file_id=_positive_int(file_record.get("id")),
                storage_ref=_first_text(file_record.get("storage_path")),
                mime_type=_first_text(file_record.get("mime_type"), image.content_type),
                width=width,
                height=height,
                bytes=stored_bytes,
                backend_metadata=backend_metadata,
                batch_id=batch_id if attempt_token else None,
                slot_id=slot_id if attempt_token else None,
                variant_index=variant_index if attempt_token else None,
                attempt_token=attempt_token,
                validate_authority=(lambda: self._require_current_job_lease(job, user_id=user_id)) if job else None,
            ) or item
        except VNAssetGenerationError:
            if attempt_token is not None and file_record is not None:
                await self._cleanup_cancelled_variant_storage(
                    batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                    user_id=user_id, pack_id=pack_id, file_record=file_record,
                )
            raise
        except Exception as exc:
            if int(batch.get("recipe_version") or 0) == 0:
                self.repo.delete_item(item_id)
                raise
            if file_record is not None:
                await self._cleanup_cancelled_variant_storage(
                    batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                    user_id=user_id, pack_id=pack_id, file_record=file_record,
                )
            raise VNAssetGenerationError(
                "vn_asset_storage_handoff_retryable", retryable=True,
                batch_id=batch_id, slot_id=slot_id, item_id=item_id,
            ) from exc
        if int(batch.get("recipe_version") or 0) == 1:
            try:
                if job is not None:
                    await self.repo.run_worker_replay_operation(
                        partial(self._require_current_job_lease, job, user_id=user_id)
                    )
                item = await self.repo.run_worker_replay_operation(partial(
                    self.repo.complete_variant,
                    batch_id=batch_id, slot_id=slot_id,
                    variant_index=variant_index, item_id=item_id,
                    attempt_token=attempt_token,
                    validate_authority=(lambda: self._require_current_job_lease(job, user_id=user_id)) if job else None,
                ))
            except VNAssetGenerationError:
                await self._cleanup_cancelled_variant_storage(
                    batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                    user_id=user_id, pack_id=pack_id, file_record=file_record,
                )
                raise
            except Exception as exc:
                raise VNAssetGenerationError(
                    "vn_asset_publication_retryable", retryable=True,
                    batch_id=batch_id, slot_id=slot_id, item_id=item_id,
                ) from exc
        else:
            self.repo.mark_slot_generation_succeeded(slot_id, batch_id)
            self._record_generation_success(batch_id=batch_id)
        logger.info(
            "VN asset variant generated: pack_id={} slot_id={} item_id={} backend={}",
            pack_id,
            slot_id,
            item["id"],
            backend,
        )
        return _generated_variant_result(item, batch_id=batch_id)

    async def _cleanup_cancelled_variant_storage(
        self, *, batch_id: int, slot_id: int, variant_index: int,
        user_id: int, pack_id: int, file_record: Mapping[str, Any] | None = None,
    ) -> None:
        """Release only an owned unreferenced file orphaned by terminal cancellation.

        Recheck the current registry and VN admission before quota-aware removal.
        Detach only the exact owned hidden terminal attachment under variant
        write admission; an interrupted acknowledgement is recoverable by source.
        Lease takeover/requested cancellation without a terminal VN outcome stays
        recoverable. Referenced/foreign files and ledger/counters are unchanged.
        Keep registration/charge until unlink succeeds so terminal redelivery can
        retry physical failures. Failed unregistration conservatively retains
        charge for already-missing bytes until redelivery; failures propagate.
        """
        identity = {
            "batch_id": batch_id, "slot_id": slot_id, "variant_index": variant_index,
            "user_id": user_id, "pack_id": pack_id,
        }
        item = await self.repo.run_worker_replay_operation(partial(
            self.repo.get_cancelled_variant_storage_item, **identity,
        ))
        if item is None:
            return
        files_repo = self.generated_files_repo
        storage_service = None
        if files_repo is None:
            storage_service = await get_storage_service()
            files_repo = await storage_service.get_generated_files_repo()
        if file_record is None:
            file_record = await _maybe_await(files_repo.get_file_by_source_ref(
                user_id=user_id, source_feature=SOURCE_FEATURE_VN_ASSETS,
                source_ref=vn_asset_source_ref(int(item["id"])),
            ))
        if file_record is None or _positive_int(file_record.get("id")) is None:
            return
        file_id = int(file_record["id"])
        current = await _maybe_await(files_repo.get_file_by_id(file_id))
        try:
            matches = current is not None and generated_file_matches_vn_asset(
                current, user_id=user_id, item_id=int(item["id"]),
            )
        except (TypeError, ValueError):
            matches = False
        if not matches or current is None:
            return
        storage_path = str(current.get("storage_path") or "")
        resolve_vn_asset_storage_path(user_id=user_id, storage_path=storage_path)
        if await self.repo.run_worker_replay_operation(partial(
            self.repo.get_cancelled_variant_storage_item, **identity, generated_file_id=file_id,
        )) is None:
            return
        unregister = self.unregister_generated_file
        if unregister is None:
            storage_service = storage_service or await get_storage_service()
            unregister = storage_service.unregister_generated_file
        await asyncio.to_thread(unlink_vn_asset_storage_file, user_id=user_id, storage_path=storage_path)
        if not await _maybe_await(unregister(file_id, hard_delete=True)):
            raise VNAssetGenerationError(
                "vn_asset_cancelled_storage_cleanup_retryable", retryable=True,
                batch_id=batch_id, item_id=int(item["id"]),
            )

    def _resolve_backend(self, requested_backend: Any) -> str:
        requested_backend = _first_text(requested_backend)
        resolver = getattr(self.image_registry, "resolve_backend", None)
        backend = resolver(requested_backend) if callable(resolver) else requested_backend
        if not backend:
            raise VNAssetGenerationError("image_backend_unavailable", requested_backend=requested_backend)
        return str(backend)

    def _resolve_slot_execution(self, slot: Mapping[str, Any], config: Any) -> dict[str, Any]:
        """Pin the public backend/model selection without persisting local paths."""
        backend = self._resolve_backend(slot.get("requested_backend"))
        resolved = {
            "slot_id": slot["slot_id"],
            "backend": backend,
            "model": resolve_image_generation_model(backend, slot.get("requested_model"), config),
        }
        if backend == "stable_diffusion_cpp":
            mode, configured_path = _local_model_state(config)
            resolved["local_model_mode"] = mode
            if resolved["model"] is None:
                resolved["local_model_path_sha256"] = _path_digest(configured_path)
        return resolved

    @staticmethod
    def _execution_recipe(batch: Mapping[str, Any]) -> dict[str, Any]:
        """Read the batch's pinned execution recipe or reject an invalid snapshot."""
        return load_execution_recipe(batch["execution_recipe_json"])

    def _record_generation_success(self, *, batch_id: int) -> None:
        """Advance batch completion without reopening a terminal batch."""
        self.repo.record_batch_variant_success(batch_id)

    def _record_generation_failure(
        self, *, batch_id: int, slot_id: int, variant_index: int, error: str,
        attempt_token: str | None = None,
        job: Mapping[str, Any] | None = None,
        user_id: int | None = None,
    ) -> None:
        """Record failure through the version-specific persistence boundary.

        The asynchronous delivery handler runs this complete operation through
        the existing owning-thread boundary, including batch read/reconciliation.

        Args:
            batch_id: Batch whose persisted recipe version selects the transition.
            slot_id: Slot to reconcile for V1 or mark failed for legacy generation.
            variant_index: Exact V1 recipe variant; unused by the legacy transition.
            error: Safe persisted failure description.
            attempt_token: Current V1 claim fence, forwarded to failure admission.
            job: Optional Jobs delivery whose current lease must authorize V1 writes.
            user_id: Delivery owner required when job supplies that authority callback.

        Returns:
            None: Missing batches do nothing. V1 uses fenced variant failure admission;
            legacy generation updates slot status/error and batch status/failed count.

        Raises:
            VNAssetGenerationError: Jobs authority validation rejects a V1 transition.
            Exception: Native database or authority callback failures propagate.
        """
        batch = self.repo.get_batch(batch_id)
        if batch is None:
            return
        if int(batch.get("recipe_version") or 0) == 1:
            self.repo.fail_variant(
                batch_id=batch_id, slot_id=slot_id, variant_index=variant_index,
                error=error, attempt_token=attempt_token,
                validate_authority=(lambda: self._require_current_job_lease(job, user_id=int(user_id))) if job else None,
            )
            return
        self.repo.record_batch_variant_failure(batch_id, slot_id=slot_id, error=error)

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


def _payload_int(payload: Mapping[str, Any], key: str, *, default: int | None = None) -> int:
    try:
        return int(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        if default is not None:
            return default
        raise ValueError(f"missing_{key}") from exc


def _payload_optional_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_{key}") from exc


def _payload_text(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"missing_{key}")
    return value


_TERMINAL_BATCH_STATUSES = {"cancelled", "canceled", "completed", "failed"}


def _is_terminal_batch_status(status: Any) -> bool:
    return str(status or "").strip().lower() in _TERMINAL_BATCH_STATUSES


def _retryable_fanout_failure(batch: Mapping[str, Any]) -> bool:
    return (
        batch["status"] == "failed"
        and batch.get("enqueue_error") is not None
        and int(batch["failed_count"] or 0) == 0
    )


def _job_has_retry_remaining(job: Mapping[str, Any] | None) -> bool:
    """Leave retryable attempts active until the Jobs retry budget is exhausted."""
    if job is None:
        return False
    try:
        return int(job.get("retry_count") or 0) < int(job.get("max_retries") or 0)
    except (TypeError, ValueError):
        return False


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


def _deleted_variant_result(
    outcome: Mapping[str, Any], *, batch_id: int, slot_id: int, variant_index: int,
    user_id: int, pack_id: int,
) -> dict[str, Any]:
    """Replay an intentional deletion receipt without resurrecting assets.

    Args:
        outcome: Persisted recipe outcome and deletion receipt.
        batch_id: Original batch identity.
        slot_id: Original slot identity.
        variant_index: Original variant identity.
        user_id: Already-authorized batch owner.
        pack_id: Already-authorized pack identity.

    Returns:
        Original stable Jobs result; malformed or mismatched receipts fail closed.
    """
    try:
        receipt = outcome["deleted_item_json"]
        item = json.loads(receipt) if isinstance(receipt, str) else None
    except (ValueError, RecursionError):
        item = None
    expected = {"batch_id": batch_id, "slot_id": slot_id, "variant_index": variant_index,
                "owner_user_id": user_id, "pack_id": pack_id}
    valid = (
        outcome["outcome_status"] == "completed" and outcome.get("item_id") is None
        and isinstance(item, dict) and variant_index >= 0
        and all(type(item.get(key)) is int and item[key] == value for key, value in expected.items())
        and all(type(item.get(key)) is int and item[key] > 0 for key in ("id", "generated_file_id"))
    )
    if not valid:
        raise VNAssetGenerationError("vn_asset_recipe_item_missing", batch_id=batch_id, slot_id=slot_id)
    return _generated_variant_result(item, batch_id=batch_id)


def _generated_variant_result(item: Mapping[str, Any], *, batch_id: int) -> dict[str, Any]:
    """Return the stable Jobs result for item and batch_id.

    Missing item keys or malformed numeric IDs raise KeyError/ValueError.
    """
    return {
        "status": "draft_created",
        "pack_id": int(item["pack_id"]),
        "slot_id": int(item["slot_id"]),
        "item_id": int(item["id"]),
        "batch_id": batch_id,
        "generated_file_id": item["generated_file_id"],
    }


def _world_book_entries_for_pack(
    repo: VNAssetPacksRepository,
    pack: Mapping[str, Any],
    *, strict: bool = False,
) -> list[Any]:
    """Read selected enabled entries; strict snapshot outages raise a safe VN error.

    Unconfigured or genuinely empty books return an empty list. Legacy callers
    keep the optional logged fallback; new snapshots must not freeze failed reads.
    """
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
        if strict:
            raise VNAssetGenerationError(
                "vn_asset_world_book_context_unavailable", retryable=True,
                pack_id=pack.get("id"), operation="read_world_book_context",
            ) from None
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


def _local_model_state(config: Any) -> tuple[str, str | None]:
    diffusion_path = getattr(config, "sd_cpp_diffusion_model_path", None)
    mode = "diffusion" if diffusion_path else "model"
    raw_path = diffusion_path or getattr(config, "sd_cpp_model_path", None)
    path = str(Path(raw_path).expanduser().resolve(strict=False)) if raw_path else None
    return mode, path


def _path_digest(path: str | None) -> str | None:
    return sha256(path.encode("utf-8")).hexdigest() if path is not None else None


def _generation_shape(
    pack: Mapping[str, Any], slot: Mapping[str, Any]
) -> tuple[int | None, int | None, str, dict[str, Any]]:
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


def build_slot_recipe(
    repo: VNAssetPacksRepository,
    pack: Mapping[str, Any],
    slot: Mapping[str, Any],
    character: Mapping[str, Any],
    *, strict_world_books: bool = False,
) -> dict[str, Any]:
    """Return frozen generation parameters from pack, slot, and character rows.

    repo supplies world-book context. Invalid required row fields or prompt
    construction failures propagate. strict_world_books aborts a new snapshot
    with a safe retryable VN error on configured-book read failures; the default
    retains the optional legacy fallback. Unconfigured/empty books remain valid.
    """
    labels = _loads_json(slot.get("labels_json"), {})
    preview = build_prompt_preview(
        character=character,
        pack_style=pack.get("style_prompt"),
        pack_scenario=pack.get("scenario_notes"),
        negative_prompt=_join_prompt_parts(
            pack.get("negative_prompt"), slot.get("negative_prompt_template")
        ),
        style_lock=_loads_json(pack.get("style_lock_json"), {}),
        slot_template=slot.get("prompt_template"),
        labels=labels,
        world_book_entries=_world_book_entries_for_pack(repo, pack, strict=strict_world_books),
    )
    width, height, image_format, extra_params = _generation_shape(pack, slot)
    return {
        "prompt": preview.prompt,
        "negative_prompt": preview.negative_prompt,
        "token_estimates": preview.token_estimates,
        "omitted_source_counts": preview.omitted_source_counts,
        "warnings": list(preview.warnings),
        "requested_backend": _first_text(slot.get("backend_override"), pack.get("default_backend")),
        "model": _first_text(slot.get("model_override"), pack.get("default_model")),
        "width": width,
        "height": height,
        "format": image_format,
        "steps": _positive_int(extra_params.pop("steps", None)),
        "cfg_scale": _float_or_none(extra_params.pop("cfg_scale", None)),
        "sampler": _first_text(extra_params.pop("sampler", None)),
        "extra_params": extra_params,
        "labels": labels,
        "asset_type": str(slot["asset_type"]),
        "slot_key": slot.get("slot_key"),
        "primary_character_id": pack.get("primary_character_id"),
    }


def authored_variant_recipe(
    recipe: Mapping[str, Any], slot: Mapping[str, Any], variant_index: int,
) -> dict[str, Any]:
    """Project one frozen authored slot into the durable per-variant format.

    Args:
        recipe: Validated authored batch containing its original character ID.
        slot: Frozen slot inputs from that batch, not a current database row.
        variant_index: Recorded variant whose seed must be retained exactly.

    Returns:
        Frozen generation parameters with prompt metadata and labels preserved.

    Raises:
        ValueError: The variant is absent from the recorded slot.
        KeyError: Required authored fields are missing.
    """
    if variant_index < 0 or variant_index >= int(slot["variant_count"]):
        raise ValueError("vn_asset_recipe_variant_mismatch")
    snapshot = slot["prompt_snapshot"]
    extra_params = dict(slot["extra_params"])
    return {
        "prompt": snapshot["prompt"],
        "negative_prompt": snapshot["negative_prompt"],
        "token_estimates": snapshot.get("token_estimates", {}),
        "omitted_source_counts": snapshot.get("omitted_source_counts", {}),
        "warnings": snapshot.get("warnings", []),
        "requested_backend": slot["requested_backend"],
        "model": slot["requested_model"],
        "width": slot["width"], "height": slot["height"], "format": slot["format"],
        "steps": _positive_int(extra_params.pop("steps", None)),
        "cfg_scale": _float_or_none(extra_params.pop("cfg_scale", None)),
        "sampler": _first_text(extra_params.pop("sampler", None)),
        "extra_params": extra_params,
        "seed": slot["seeds"][variant_index],
        "labels": dict(slot["labels"]),
        "asset_type": slot["asset_type"], "slot_key": slot["slot_key"],
        "primary_character_id": recipe["primary_character_id"],
    }


def variant_seed(slot: Mapping[str, Any], variant_index: int) -> int | None:
    """Return slot's seed/base_seed plus variant_index, or None without a valid seed.

    Missing or malformed optional seed values are treated as unconfigured.
    """
    seed_policy = _loads_json(slot.get("seed_policy_json"), {})
    seed = _positive_int(seed_policy.get("seed"))
    if seed is not None:
        return seed + variant_index
    base_seed = _positive_int(seed_policy.get("base_seed"))
    if base_seed is not None:
        return base_seed + variant_index
    return None


def _replay_file_bytes_present(record: dict[str, Any], *, user_id: int) -> bool:
    """Check contained replay bytes against size and persisted checksum off-loop.

    Filesystem and invalid-path errors propagate to the typed replay boundary.
    """
    path = resolve_vn_asset_storage_path(
        user_id=user_id, storage_path=str(record.get("storage_path") or ""),
    )
    size = generated_file_size_bytes(record)
    return generated_file_bytes_match(path, expected_size=size, checksum=record.get("checksum"))


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


def _export_options(value: Any) -> VNPackExportOptions:
    options = value if isinstance(value, Mapping) else {}
    return VNPackExportOptions(
        include_character_payload=_bool_option(options.get("include_character_payload"), default=False),
        include_world_book_payloads=_bool_option(options.get("include_world_book_payloads"), default=False),
        include_full_provenance=_bool_option(options.get("include_full_provenance"), default=False),
        strict=_bool_option(options.get("strict"), default=False),
        warn_for_sharing=_bool_option(options.get("warn_for_sharing"), default=True),
    )


def _bool_option(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _job_id(job: Mapping[str, Any] | None) -> str | None:
    if not job:
        return None
    return _first_text(job.get("id"), job.get("uuid"))
