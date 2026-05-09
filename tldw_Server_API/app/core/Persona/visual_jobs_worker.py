from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Image_Generation.adapter_registry import get_registry
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int as _coerce_int
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env as _jobs_manager
from tldw_Server_API.app.core.Persona.visual_jobs import (
    PERSONA_VISUALS_DOMAIN,
    PERSONA_VISUAL_GENERATE_CANDIDATE_JOB_TYPE,
    persona_visual_generation_queue,
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

        pack = self._db.get_persona_visual_pack(
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

        service = PersonaVisualService(self._db)
        asset = service.create_generated_asset(
            persona_id=persona_id,
            user_id=user_id,
            pack_id=pack_id,
            content=result.content,
            mime_type=result.content_type or "image/png",
            original_filename=f"generated-{target_state or 'candidate'}-{job_id or 'job'}.png",
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
        return {
            "status": "candidate_created",
            "candidate_id": str(candidate["id"]),
            "asset_id": asset_id,
            "pack_id": pack_id,
            "persona_id": persona_id,
        }


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


__all__ = [
    "PersonaVisualGenerationWorker",
    "run_persona_visual_generation_worker",
]
