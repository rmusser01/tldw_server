"""Owner-bound Document Insights prompt preparation and cache identity."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from tldw_Server_API.app.core.Prompt_Management.service_prompts import resolve_service_prompt

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase


def prepare_document_insights_prompt(database: PromptsDatabase) -> tuple[str, str]:
    """Return the assembled system message and its content-based cache fingerprint.

    The caller supplies authenticated-owner storage and runs this function on a
    worker. Read and close its connection on that same worker, including failed
    reads. Fixed output instructions are never part of the editable guidance.
    """
    try:
        prompt = resolve_service_prompt(database, "media.document.insights")
    finally:
        database.close_connection()

    system_prompt = (
        f"{prompt.parts['analysis_guidance']}\n\nReturn JSON with this structure:\n"
        '{"insights": [{"category": "...", "title": "...", "content": "..."}]}\n\n'
        f"{prompt.parts['presentation_guidance']}\n- Return ONLY valid JSON, no other text\n"
    )
    return system_prompt, hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
