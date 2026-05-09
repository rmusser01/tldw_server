from __future__ import annotations

import os
from typing import Any


PERSONA_VISUALS_DOMAIN = "persona_visuals"
PERSONA_VISUAL_GENERATE_CANDIDATE_JOB_TYPE = "persona_visual_generate_candidate"


def persona_visual_generation_queue() -> str:
    return (os.getenv("PERSONA_VISUAL_GENERATION_JOBS_QUEUE") or "generation").strip() or "generation"


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
        idempotency_key=(
            f"{PERSONA_VISUALS_DOMAIN}:{user_id}:{persona_id}:{pack_id}:"
            f"{normalized_target_state or 'pack'}"
        ),
        max_retries=1,
    )


__all__ = [
    "PERSONA_VISUALS_DOMAIN",
    "PERSONA_VISUAL_GENERATE_CANDIDATE_JOB_TYPE",
    "build_generate_candidate_payload",
    "create_generate_candidate_job",
    "persona_visual_generation_queue",
]
