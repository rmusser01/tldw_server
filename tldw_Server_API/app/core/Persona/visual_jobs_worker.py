from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
    PersonaVisualPortabilityRepository,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Image_Generation.adapter_registry import get_registry
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int as _coerce_int
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env as _jobs_manager
from tldw_Server_API.app.core.Persona.visual_jobs import (
    PERSONA_VISUALS_DOMAIN,
    PERSONA_VISUAL_GENERATE_CANDIDATE_JOB_TYPE,
    PERSONA_VISUAL_PACK_EXPORT_JOB_TYPE,
    PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE,
    PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE,
    persona_visual_generation_queue,
    persona_visual_portability_queue,
)
from tldw_Server_API.app.core.Persona.visual_portability.exporter import (
    PersonaVisualPackExporter,
)
from tldw_Server_API.app.core.Persona.visual_portability.importer import (
    PersonaVisualPackImporter,
)
from tldw_Server_API.app.core.Persona.visual_portability.models import (
    PersonaVisualPackExportOptions,
)
from tldw_Server_API.app.core.Persona.visual_portability.preview import (
    PersonaVisualPackImportPreviewer,
)
from tldw_Server_API.app.core.Persona.visual_service import PersonaVisualService


class PersonaVisualGenerationWorker:
    """Create review-gated persona visual candidates from Jobs payloads."""

    def __init__(self, *, db: CharactersRAGDB, image_registry: Any | None = None) -> None:
        self._db = db
        self._image_registry = image_registry or get_registry()

    async def handle_job_async(self, job: dict[str, Any]) -> dict[str, Any]:
        job_type = str(job.get("job_type") or "")
        if job_type != PERSONA_VISUAL_GENERATE_CANDIDATE_JOB_TYPE:
            raise ValueError(f"unsupported_persona_visual_job_type:{job_type}")

        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        user_id = str(payload.get("user_id") or "").strip()
        persona_id = str(payload.get("persona_id") or "").strip()
        pack_id = str(payload.get("pack_id") or "").strip()
        prompt = str(payload.get("prompt") or "").strip()
        target_state = str(payload.get("target_state") or "").strip() or None
        requested_backend = str(payload.get("backend") or "").strip() or None
        if not user_id or not persona_id or not pack_id or not prompt:
            raise ValueError("invalid_persona_visual_generation_payload")

        pack = await asyncio.to_thread(
            self._db.get_persona_visual_pack,
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
        )
        if not pack:
            raise ValueError("persona_visual_pack_not_found")

        backend = self._image_registry.resolve_backend(requested_backend)
        if not backend:
            raise ValueError("image_backend_unavailable")
        adapter = self._image_registry.get_adapter(backend)
        if adapter is None:
            raise ValueError("image_adapter_unavailable")

        job_id = str(job.get("id") or "")
        request = ImageGenRequest(
            backend=backend,
            prompt=prompt,
            negative_prompt=None,
            width=1024,
            height=1024,
            steps=None,
            cfg_scale=None,
            seed=None,
            sampler=None,
            model=None,
            format="png",
            extra_params={},
            request_id=f"persona_visuals:{persona_id}:{pack_id}:{job_id}",
        )
        result = await asyncio.to_thread(adapter.generate, request)

        asset, candidate = await asyncio.to_thread(
            self._persist_generated_candidate,
            persona_id=persona_id,
            user_id=user_id,
            pack_id=pack_id,
            content=result.content,
            mime_type=result.content_type or "image/png",
            original_filename=f"generated-{target_state or 'candidate'}-{job_id or 'job'}.png",
            target_state=target_state,
            job_id=job_id,
            prompt=prompt,
        )
        asset_id = str(asset["id"])
        return {
            "status": "candidate_created",
            "candidate_id": str(candidate["id"]),
            "asset_id": asset_id,
            "pack_id": pack_id,
            "persona_id": persona_id,
        }

    def _persist_generated_candidate(
        self,
        *,
        persona_id: str,
        user_id: str,
        pack_id: str,
        content: bytes,
        mime_type: str,
        original_filename: str,
        target_state: str | None,
        job_id: str,
        prompt: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        service = PersonaVisualService(self._db)
        asset = service.create_generated_asset(
            persona_id=persona_id,
            user_id=user_id,
            pack_id=pack_id,
            content=content,
            mime_type=mime_type,
            original_filename=original_filename,
        )
        asset_id = str(asset["id"])
        animation_id = f"generated-{target_state or 'candidate'}-{asset_id[:8]}"
        proposed_patch: dict[str, Any] = {
            "states": (
                {target_state: {"animation_id": animation_id}}
                if target_state
                else {}
            ),
            "animations": {
                animation_id: {
                    "asset_ids": [asset_id],
                    "frame_rate": 1,
                    "loop": True,
                    "alignment": {"x": 0.5, "y": 1.0},
                }
            },
            "authored_triggers": [],
        }
        candidate = self._db.create_persona_visual_candidate(
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
            job_id=job_id,
            proposed_manifest_patch=proposed_patch,
            generated_asset_ids=[asset_id],
            prompt=prompt,
        )
        return asset, candidate


class PersonaVisualPortabilityWorker:
    """Run persona visual pack export and import-preview Jobs."""

    def __init__(
        self,
        *,
        db: CharactersRAGDB,
        repo: PersonaVisualPortabilityRepository | None = None,
        export_staging_root: Path | None = None,
    ) -> None:
        self._db = db
        self._repo = repo or PersonaVisualPortabilityRepository.initialized(db)
        self._export_staging_root = Path(export_staging_root) if export_staging_root is not None else None

    async def handle_job_async(self, job: dict[str, Any]) -> dict[str, Any]:
        job_type = str(job.get("job_type") or "")
        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        if job_type == PERSONA_VISUAL_PACK_EXPORT_JOB_TYPE:
            return await self.handle_export_pack(payload, job=job)
        if job_type == PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE:
            return await self.handle_import_preview(payload, job=job)
        if job_type == PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE:
            return await self.handle_import_commit(payload, job=job)
        raise ValueError(f"unsupported_persona_visual_portability_job_type:{job_type}")

    async def handle_export_pack(
        self,
        payload: dict[str, Any],
        *,
        job: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        user_id = _payload_text(payload, "user_id")
        persona_id = _payload_text(payload, "persona_id")
        pack_id = _payload_text(payload, "pack_id")
        portability_job_id = _payload_text(payload, "portability_job_id", default="")
        if not user_id or not persona_id or not pack_id:
            raise ValueError("invalid_persona_visual_pack_export_payload")

        portability_job = (
            self._repo.get_portability_job(portability_job_id, owner_user_id=user_id)
            if portability_job_id
            else self._repo.get_portability_job_by_job_id(str((job or {}).get("id") or ""), owner_user_id=user_id)
        )
        if (
            portability_job is None
            or str(portability_job.get("operation") or "") != "export"
            or str(portability_job.get("persona_id") or "") != persona_id
            or str(portability_job.get("pack_id") or "") != pack_id
        ):
            raise ValueError("persona_visual_pack_portability_job_not_found")

        job_id = str(portability_job["job_id"])
        self._repo.update_portability_job(
            job_id,
            {"status": "processing", "stage": "collecting_metadata", "progress": {"pack_id": pack_id}},
            owner_user_id=user_id,
        )

        def _progress(stage: str, progress: dict[str, Any]) -> None:
            self._repo.update_portability_job(
                job_id,
                {"status": "processing", "stage": stage, "progress": progress},
                owner_user_id=user_id,
            )

        try:
            exporter = PersonaVisualPackExporter(
                db=self._db,
                user_id=user_id,
                staging_root=self._resolve_export_staging_root(user_id),
            )
            result = await asyncio.to_thread(
                lambda: exporter.export_pack(
                    persona_id=persona_id,
                    pack_id=pack_id,
                    options=_export_options(payload.get("options")),
                    progress=_progress,
                )
            )
        except Exception as exc:
            self._repo.update_portability_job(
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
        self._repo.update_portability_job(
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
            "persona_id": persona_id,
            "pack_id": pack_id,
            "portability_job_id": str(portability_job["id"]),
            "archive_path": str(result.archive_path),
            "archive_sha256": result.archive_sha256,
            "canonical_payload_fingerprint": result.canonical_payload_fingerprint,
            "file_size_bytes": result.file_size_bytes,
            "warnings": result.warnings,
        }

    async def handle_import_preview(
        self,
        payload: dict[str, Any],
        *,
        job: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        user_id = _payload_text(payload, "user_id")
        preview_id = _payload_text(payload, "preview_id")
        if not user_id or not preview_id:
            raise ValueError("invalid_persona_visual_import_preview_payload")

        preview = self._repo.get_import_preview(preview_id, owner_user_id=user_id)
        if preview is None:
            raise ValueError("persona_visual_pack_import_preview_not_found")

        job_id = str(preview.get("job_id") or (job or {}).get("id") or "")
        portability_job = self._repo.get_portability_job_by_job_id(job_id, owner_user_id=user_id)
        archive_path = Path(
            _payload_text(payload, "archive_path", default=str(preview.get("archive_path") or ""))
        )
        if not archive_path.is_file():
            raise ValueError("persona_visual_pack_import_archive_not_found")

        self._repo.update_import_preview(
            preview_id,
            {"status": "processing", "stage": "validating_archive", "archive_path": str(archive_path)},
            owner_user_id=user_id,
        )
        if portability_job is not None:
            self._repo.update_portability_job(
                job_id,
                {"status": "processing", "stage": "validating_archive"},
                owner_user_id=user_id,
            )

        def _progress(stage: str, progress: dict[str, Any]) -> None:
            if portability_job is None:
                return
            self._repo.update_portability_job(
                job_id,
                {"status": "processing", "stage": stage, "progress": progress},
                owner_user_id=user_id,
            )

        target_persona_id = _payload_text(
            payload,
            "target_persona_id",
            default=str(preview.get("target_persona_id") or ""),
        ) or None
        try:
            result = await asyncio.to_thread(
                lambda: PersonaVisualPackImportPreviewer().create_preview(
                    archive_path=archive_path,
                    owner_user_id=user_id,
                    target_persona_id=target_persona_id,
                    progress=_progress,
                )
            )
        except Exception as exc:
            self._repo.update_import_preview(
                preview_id,
                {
                    "status": "failed",
                    "stage": "failed",
                    "error_code": "import_preview_failed",
                    "error_message": str(exc),
                },
                owner_user_id=user_id,
            )
            if portability_job is not None:
                self._repo.update_portability_job(
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
        self._repo.update_import_preview(
            preview_id,
            {
                "status": "completed",
                "stage": "completed",
                "archive_sha256": result["archive_sha256"],
                "canonical_payload_fingerprint": result["canonical_payload_fingerprint"],
                "schema_version": result["schema_version"],
                "bundle_summary": result["bundle_summary"],
                "validation_warnings": result["validation_warnings"],
                "conflicts": result["conflicts"],
                "proposed_plan": result["proposed_plan"],
                "quota_estimate": result["quota_estimate"],
                "required_choices": result["required_choices"],
                "target_warnings": result["target_warnings"],
                "expires_at": expires_at,
            },
            owner_user_id=user_id,
        )
        if portability_job is not None:
            self._repo.update_portability_job(
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
        payload: dict[str, Any],
        *,
        job: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        user_id = _payload_text(payload, "user_id")
        preview_id = _payload_text(payload, "preview_id")
        portability_job_id = _payload_text(payload, "portability_job_id", default="")
        target_persona_id = _payload_text(payload, "target_persona_id")
        trust_mode = _payload_text(payload, "trust_mode", default="untrusted_import")
        target_mode = _payload_text(payload, "target_mode", default="create_new")
        if not user_id or not preview_id or not target_persona_id:
            raise ValueError("invalid_persona_visual_import_commit_payload")

        preview = self._repo.get_import_preview(preview_id, owner_user_id=user_id)
        if preview is None:
            raise ValueError("persona_visual_pack_import_preview_not_found")
        portability_job = (
            self._repo.get_portability_job(portability_job_id, owner_user_id=user_id)
            if portability_job_id
            else self._repo.get_portability_job_by_job_id(str((job or {}).get("id") or ""), owner_user_id=user_id)
        )
        if (
            portability_job is None
            or str(portability_job.get("operation") or "") != "import_commit"
            or str(portability_job.get("preview_id") or "") != preview_id
            or str(portability_job.get("persona_id") or "") != target_persona_id
        ):
            raise ValueError("persona_visual_pack_import_commit_job_not_found")

        job_id = str(portability_job["job_id"])
        self._repo.update_portability_job(
            job_id,
            {"status": "processing", "stage": "revalidating_preview"},
            owner_user_id=user_id,
        )

        def _progress(stage: str, progress: dict[str, Any]) -> None:
            self._repo.update_portability_job(
                job_id,
                {"status": "processing", "stage": stage, "progress": progress},
                owner_user_id=user_id,
            )

        try:
            result = await asyncio.to_thread(
                lambda: PersonaVisualPackImporter(
                    db=self._db,
                    repo=self._repo,
                    user_id=user_id,
                ).import_preview(
                    preview_id=preview_id,
                    target_persona_id=target_persona_id,
                    trust_mode=trust_mode,
                    target_mode=target_mode,
                    progress=_progress,
                )
            )
        except Exception as exc:
            self._repo.update_portability_job(
                job_id,
                {
                    "status": "failed",
                    "stage": "failed",
                    "error_code": "import_commit_failed",
                    "error_message": str(exc),
                },
                owner_user_id=user_id,
            )
            raise

        self._repo.update_import_preview(
            preview_id,
            {"status": "imported", "stage": "imported"},
            owner_user_id=user_id,
        )
        self._repo.update_portability_job(
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "persona_id": target_persona_id,
                "pack_id": result["pack_id"],
                "progress": {
                    "pack_id": result["pack_id"],
                    "asset_count": len(result.get("created_records", {}).get("asset_ids", [])),
                },
            },
            owner_user_id=user_id,
        )
        return {
            **result,
            "portability_job_id": str(portability_job["id"]),
            "job_id": job_id,
        }

    def _resolve_export_staging_root(self, user_id: str) -> Path:
        if self._export_staging_root is not None:
            return self._export_staging_root
        configured = (os.getenv("PERSONA_VISUAL_PACK_EXPORT_STAGING_ROOT") or "").strip()
        if configured:
            return Path(configured) / str(user_id)
        return DatabasePaths.get_user_temp_outputs_dir(user_id) / "persona_visual_packs"


async def run_persona_visual_generation_worker(stop_event: asyncio.Event | None = None) -> None:
    worker_id = (os.getenv("PERSONA_VISUAL_GENERATION_WORKER_ID") or f"persona-visuals-{os.getpid()}").strip()
    queue = persona_visual_generation_queue()
    lease_seconds = _coerce_int(
        os.getenv("PERSONA_VISUAL_GENERATION_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"),
        60,
    )
    renew_jitter = _coerce_int(
        os.getenv("PERSONA_VISUAL_GENERATION_JOBS_RENEW_JITTER_SECONDS")
        or os.getenv("JOBS_LEASE_RENEW_JITTER_SECONDS"),
        5,
    )
    renew_threshold = _coerce_int(
        os.getenv("PERSONA_VISUAL_GENERATION_JOBS_RENEW_THRESHOLD_SECONDS")
        or os.getenv("JOBS_LEASE_RENEW_THRESHOLD_SECONDS"),
        10,
    )
    cfg = WorkerConfig(
        domain=PERSONA_VISUALS_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        renew_jitter_seconds=renew_jitter,
        renew_threshold_seconds=renew_threshold,
    )
    sdk = WorkerSDK(_jobs_manager(), cfg)
    stop_watcher: asyncio.Task[None] | None = None

    if stop_event is not None:
        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_watcher = asyncio.create_task(_watch_stop())

    async def _handler(job: dict[str, Any]) -> dict[str, Any]:
        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        user_id = str(payload.get("user_id") or "").strip()
        if not user_id:
            raise ValueError("invalid_persona_visual_generation_payload")
        db = CharactersRAGDB(
            DatabasePaths.get_chacha_db_path(user_id),
            client_id="persona-visual-generation-worker",
        )
        try:
            return await PersonaVisualGenerationWorker(db=db).handle_job_async(job)
        finally:
            db.close_connection()

    logger.info("Persona visual generation worker starting: queue={} worker_id={}", queue, worker_id)
    try:
        await sdk.run(handler=_handler)
    finally:
        if stop_watcher is not None and not stop_watcher.done():
            stop_watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_watcher


async def run_persona_visual_portability_worker(stop_event: asyncio.Event | None = None) -> None:
    worker_id = (os.getenv("PERSONA_VISUAL_PORTABILITY_WORKER_ID") or f"persona-visual-portability-{os.getpid()}").strip()
    queue = persona_visual_portability_queue()
    lease_seconds = _coerce_int(
        os.getenv("PERSONA_VISUAL_PORTABILITY_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"),
        60,
    )
    renew_jitter = _coerce_int(
        os.getenv("PERSONA_VISUAL_PORTABILITY_JOBS_RENEW_JITTER_SECONDS")
        or os.getenv("JOBS_LEASE_RENEW_JITTER_SECONDS"),
        5,
    )
    renew_threshold = _coerce_int(
        os.getenv("PERSONA_VISUAL_PORTABILITY_JOBS_RENEW_THRESHOLD_SECONDS")
        or os.getenv("JOBS_LEASE_RENEW_THRESHOLD_SECONDS"),
        10,
    )
    cfg = WorkerConfig(
        domain=PERSONA_VISUALS_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        renew_jitter_seconds=renew_jitter,
        renew_threshold_seconds=renew_threshold,
    )
    sdk = WorkerSDK(_jobs_manager(), cfg)
    stop_watcher: asyncio.Task[None] | None = None

    if stop_event is not None:
        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_watcher = asyncio.create_task(_watch_stop())

    async def _handler(job: dict[str, Any]) -> dict[str, Any]:
        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        user_id = str(payload.get("user_id") or "").strip()
        if not user_id:
            raise ValueError("invalid_persona_visual_portability_payload")
        db = CharactersRAGDB(
            DatabasePaths.get_chacha_db_path(user_id),
            client_id="persona-visual-portability-worker",
        )
        try:
            return await PersonaVisualPortabilityWorker(db=db).handle_job_async(job)
        finally:
            db.close_connection()

    logger.info("Persona visual portability worker starting: queue={} worker_id={}", queue, worker_id)
    try:
        await sdk.run(handler=_handler)
    finally:
        if stop_watcher is not None and not stop_watcher.done():
            stop_watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_watcher


def _payload_text(payload: dict[str, Any], key: str, *, default: str | None = None) -> str:
    value = payload.get(key, default)
    if value is None:
        return ""
    return str(value).strip()


def _export_options(value: Any) -> PersonaVisualPackExportOptions:
    options = value if isinstance(value, dict) else {}
    return PersonaVisualPackExportOptions(
        strict=_bool_option(options.get("strict"), default=False),
        include_full_provenance=_bool_option(options.get("include_full_provenance"), default=False),
        warn_for_sharing=_bool_option(options.get("warn_for_sharing"), default=True),
    )


def _bool_option(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


__all__ = [
    "PersonaVisualGenerationWorker",
    "PersonaVisualPortabilityWorker",
    "run_persona_visual_generation_worker",
    "run_persona_visual_portability_worker",
]
