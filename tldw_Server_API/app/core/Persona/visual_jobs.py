from __future__ import annotations

import hashlib
import json
import os
from typing import Any


PERSONA_VISUALS_DOMAIN = "persona_visuals"
PERSONA_VISUAL_GENERATE_CANDIDATE_JOB_TYPE = "persona_visual_generate_candidate"
PERSONA_VISUAL_PACK_EXPORT_JOB_TYPE = "persona_visual_pack_export"
PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE = "persona_visual_pack_import_preview"
PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE = "persona_visual_pack_import_commit"


def persona_visual_generation_queue() -> str:
    return (os.getenv("PERSONA_VISUAL_GENERATION_JOBS_QUEUE") or "generation").strip() or "generation"


def persona_visual_portability_queue() -> str:
    return (os.getenv("PERSONA_VISUAL_PORTABILITY_JOBS_QUEUE") or "default").strip() or "default"


def build_generate_candidate_payload(
    *,
    user_id: str,
    persona_id: str,
    pack_id: str,
    prompt: str,
    target_state: str | None,
    backend: str | None,
) -> dict[str, Any]:
    return {
        "user_id": str(user_id),
        "persona_id": str(persona_id),
        "pack_id": str(pack_id),
        "prompt": str(prompt),
        "target_state": str(target_state).strip() if target_state else None,
        "backend": str(backend).strip() if backend else None,
    }


def build_visual_pack_export_payload(
    *,
    user_id: str,
    persona_id: str,
    pack_id: str,
    portability_job_id: str,
    request_id: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "user_id": str(user_id),
        "persona_id": str(persona_id),
        "pack_id": str(pack_id),
        "portability_job_id": str(portability_job_id),
        "request_id": str(request_id),
        "options": dict(options or {}),
    }


def build_visual_pack_import_preview_payload(
    *,
    user_id: str,
    preview_id: str,
    archive_path: str,
    request_id: str,
    target_persona_id: str | None = None,
) -> dict[str, Any]:
    return {
        "user_id": str(user_id),
        "preview_id": str(preview_id),
        "archive_path": str(archive_path),
        "request_id": str(request_id),
        "target_persona_id": str(target_persona_id).strip() if target_persona_id else None,
    }


def build_visual_pack_import_commit_payload(
    *,
    user_id: str,
    preview_id: str,
    portability_job_id: str,
    request_id: str,
    target_persona_id: str,
    trust_mode: str,
    target_mode: str = "create_new",
    target_pack_id: str | None = None,
    title: str | None = None,
    conflict_choice_explicit: bool = False,
) -> dict[str, Any]:
    payload = {
        "user_id": str(user_id),
        "preview_id": str(preview_id),
        "portability_job_id": str(portability_job_id),
        "request_id": str(request_id),
        "target_persona_id": str(target_persona_id),
        "trust_mode": str(trust_mode),
        "target_mode": str(target_mode or "create_new"),
    }
    if target_pack_id:
        payload["target_pack_id"] = str(target_pack_id).strip()
    if title:
        payload["title"] = str(title).strip()
    if conflict_choice_explicit:
        payload["conflict_choice_explicit"] = True
    return payload


def visual_pack_export_group(
    *,
    user_id: str,
    persona_id: str,
    pack_id: str,
    request_id: str,
) -> str:
    return (
        f"{PERSONA_VISUALS_DOMAIN}:user:{user_id}:persona:{persona_id}:"
        f"pack:{pack_id}:portability:export:{request_id}"
    )


def visual_pack_import_preview_group(
    *,
    user_id: str,
    preview_id: str,
    request_id: str,
) -> str:
    return (
        f"{PERSONA_VISUALS_DOMAIN}:user:{user_id}:portability:"
        f"import-preview:{preview_id}:{request_id}"
    )


def visual_pack_import_commit_group(
    *,
    user_id: str,
    preview_id: str,
    request_id: str,
) -> str:
    return (
        f"{PERSONA_VISUALS_DOMAIN}:user:{user_id}:portability:"
        f"import-commit:{preview_id}:{request_id}"
    )


def visual_pack_export_idempotency_key(
    *,
    user_id: str,
    persona_id: str,
    pack_id: str,
    request_id: str,
    options: dict[str, Any] | None = None,
) -> str:
    options_digest = hashlib.sha256(
        json.dumps(
            dict(options or {}),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return (
        f"{visual_pack_export_group(user_id=user_id, persona_id=persona_id, pack_id=pack_id, request_id=request_id)}"
        f":{options_digest}"
    )


def visual_pack_import_preview_idempotency_key(
    *,
    user_id: str,
    preview_id: str,
    archive_path: str,
    request_id: str,
) -> str:
    archive_digest = hashlib.sha256(str(archive_path).encode("utf-8")).hexdigest()[:16]
    return (
        f"{visual_pack_import_preview_group(user_id=user_id, preview_id=preview_id, request_id=request_id)}"
        f":{archive_digest}"
    )


def visual_pack_import_commit_idempotency_key(
    *,
    user_id: str,
    preview_id: str,
    request_id: str,
    trust_mode: str,
    target_mode: str = "create_new",
    target_pack_id: str | None = None,
    title: str | None = None,
    conflict_choice_explicit: bool = False,
) -> str:
    commit_digest = hashlib.sha256(
        json.dumps(
            {
                "conflict_choice_explicit": bool(conflict_choice_explicit),
                "target_mode": str(target_mode or "create_new").strip(),
                "target_pack_id": str(target_pack_id).strip() if target_pack_id else None,
                "title": str(title).strip() if title else None,
                "trust_mode": str(trust_mode).strip(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return (
        f"{visual_pack_import_commit_group(user_id=user_id, preview_id=preview_id, request_id=request_id)}"
        f":{commit_digest}"
    )


def visual_generate_candidate_idempotency_key(
    *,
    user_id: str,
    persona_id: str,
    pack_id: str,
    prompt: str,
    target_state: str | None = None,
    backend: str | None = None,
) -> str:
    normalized_target_state = str(target_state).strip() if target_state else None
    generation_digest = hashlib.sha256(
        json.dumps(
            {
                "backend": str(backend).strip() if backend else None,
                "prompt": str(prompt or "").strip(),
                "target_state": normalized_target_state,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return (
        f"{PERSONA_VISUALS_DOMAIN}:{user_id}:{persona_id}:{pack_id}:"
        f"{normalized_target_state or 'pack'}:{generation_digest}"
    )


def create_generate_candidate_job(
    jobs_manager: Any,
    *,
    user_id: str,
    persona_id: str,
    pack_id: str,
    prompt: str,
    target_state: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    normalized_prompt = str(prompt or "").strip()
    if not normalized_prompt:
        raise ValueError("prompt is required")
    normalized_target_state = str(target_state).strip() if target_state else None
    payload = build_generate_candidate_payload(
        user_id=user_id,
        persona_id=persona_id,
        pack_id=pack_id,
        prompt=normalized_prompt,
        target_state=normalized_target_state,
        backend=backend,
    )
    return jobs_manager.create_job(
        domain=PERSONA_VISUALS_DOMAIN,
        queue=persona_visual_generation_queue(),
        job_type=PERSONA_VISUAL_GENERATE_CANDIDATE_JOB_TYPE,
        payload=payload,
        owner_user_id=str(user_id),
        idempotency_key=visual_generate_candidate_idempotency_key(
            user_id=user_id,
            persona_id=persona_id,
            pack_id=pack_id,
            prompt=normalized_prompt,
            target_state=normalized_target_state,
            backend=backend,
        ),
        max_retries=1,
    )


def create_visual_pack_export_job(
    jobs_manager: Any,
    *,
    user_id: str,
    persona_id: str,
    pack_id: str,
    portability_job_id: str,
    request_id: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return jobs_manager.create_job(
        domain=PERSONA_VISUALS_DOMAIN,
        queue=persona_visual_portability_queue(),
        job_type=PERSONA_VISUAL_PACK_EXPORT_JOB_TYPE,
        payload=build_visual_pack_export_payload(
            user_id=user_id,
            persona_id=persona_id,
            pack_id=pack_id,
            portability_job_id=portability_job_id,
            request_id=request_id,
            options=options,
        ),
        owner_user_id=str(user_id),
        batch_group=visual_pack_export_group(
            user_id=user_id,
            persona_id=persona_id,
            pack_id=pack_id,
            request_id=request_id,
        ),
        idempotency_key=visual_pack_export_idempotency_key(
            user_id=user_id,
            persona_id=persona_id,
            pack_id=pack_id,
            request_id=request_id,
            options=options,
        ),
        max_retries=2,
    )


def create_visual_pack_import_preview_job(
    jobs_manager: Any,
    *,
    user_id: str,
    preview_id: str,
    archive_path: str,
    request_id: str,
    target_persona_id: str | None = None,
) -> dict[str, Any]:
    return jobs_manager.create_job(
        domain=PERSONA_VISUALS_DOMAIN,
        queue=persona_visual_portability_queue(),
        job_type=PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE,
        payload=build_visual_pack_import_preview_payload(
            user_id=user_id,
            preview_id=preview_id,
            archive_path=archive_path,
            request_id=request_id,
            target_persona_id=target_persona_id,
        ),
        owner_user_id=str(user_id),
        batch_group=visual_pack_import_preview_group(
            user_id=user_id,
            preview_id=preview_id,
            request_id=request_id,
        ),
        idempotency_key=visual_pack_import_preview_idempotency_key(
            user_id=user_id,
            preview_id=preview_id,
            archive_path=archive_path,
            request_id=request_id,
        ),
        max_retries=2,
    )


def create_visual_pack_import_commit_job(
    jobs_manager: Any,
    *,
    user_id: str,
    preview_id: str,
    portability_job_id: str,
    request_id: str,
    target_persona_id: str,
    trust_mode: str,
    target_mode: str = "create_new",
    target_pack_id: str | None = None,
    title: str | None = None,
    conflict_choice_explicit: bool = False,
) -> dict[str, Any]:
    return jobs_manager.create_job(
        domain=PERSONA_VISUALS_DOMAIN,
        queue=persona_visual_portability_queue(),
        job_type=PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE,
        payload=build_visual_pack_import_commit_payload(
            user_id=user_id,
            preview_id=preview_id,
            portability_job_id=portability_job_id,
            request_id=request_id,
            target_persona_id=target_persona_id,
            trust_mode=trust_mode,
            target_mode=target_mode,
            target_pack_id=target_pack_id,
            title=title,
            conflict_choice_explicit=conflict_choice_explicit,
        ),
        owner_user_id=str(user_id),
        batch_group=visual_pack_import_commit_group(
            user_id=user_id,
            preview_id=preview_id,
            request_id=request_id,
        ),
        idempotency_key=visual_pack_import_commit_idempotency_key(
            user_id=user_id,
            preview_id=preview_id,
            request_id=request_id,
            trust_mode=trust_mode,
            target_mode=target_mode,
            target_pack_id=target_pack_id,
            title=title,
            conflict_choice_explicit=conflict_choice_explicit,
        ),
        max_retries=2,
    )


__all__ = [
    "PERSONA_VISUALS_DOMAIN",
    "PERSONA_VISUAL_GENERATE_CANDIDATE_JOB_TYPE",
    "PERSONA_VISUAL_PACK_EXPORT_JOB_TYPE",
    "PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE",
    "PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE",
    "build_generate_candidate_payload",
    "build_visual_pack_export_payload",
    "build_visual_pack_import_commit_payload",
    "build_visual_pack_import_preview_payload",
    "create_visual_pack_import_commit_job",
    "create_visual_pack_export_job",
    "create_visual_pack_import_preview_job",
    "create_generate_candidate_job",
    "persona_visual_generation_queue",
    "persona_visual_portability_queue",
    "visual_generate_candidate_idempotency_key",
    "visual_pack_export_group",
    "visual_pack_export_idempotency_key",
    "visual_pack_import_commit_group",
    "visual_pack_import_commit_idempotency_key",
    "visual_pack_import_preview_group",
    "visual_pack_import_preview_idempotency_key",
]
