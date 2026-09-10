"""Resolve request-bound instructions for synchronous web article summarization.

Read saved prompts from caller-supplied owner storage and close the lookup
connection on its worker, including failed reads. Engine defaults stay local to
the scraping consumers rather than becoming request overrides.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from tldw_Server_API.app.core.Prompt_Management.service_prompts import resolve_service_prompt

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase


async def resolve_web_summary_overrides(
    get_prompts_db: Callable[[], Awaitable[PromptsDatabase]],
    *,
    enabled: bool,
    system_prompt: str | None,
    custom_prompt: str | None,
) -> Mapping[str, str] | None:
    """Freeze explicit/saved parts while leaving absent engine defaults untouched.

    Only synchronous HTTP callers supply owner-bound storage. No lookup is
    needed for disabled summarization or a complete pair of explicit parts.

    Args:
        get_prompts_db: Async factory bound to the authenticated request owner.
            Its database is read and its worker connection closed in one worker.
        enabled: Whether this request enables summarization.
        system_prompt: Explicit system instructions; None means absent.
        custom_prompt: Explicit user instructions; None means absent. Empty
            strings in either part remain explicit overrides.

    Returns:
        None when summarization is disabled; otherwise an immutable mapping of
        saved and explicit system/user parts, with explicit parts taking priority.
        Missing keys leave the corresponding scraping engine default untouched.

    Raises:
        ServicePromptCorruptOverride: Saved instructions fail validation.
        Exception: Database acquisition, lookup, or cleanup errors propagate.
    """
    if not enabled:
        return None
    parts = {key: value for key, value in (("system", system_prompt), ("user", custom_prompt)) if value is not None}
    if len(parts) < 2:
        database = await get_prompts_db()

        def load_saved() -> dict[str, str]:
            """Read and close on one worker; packaged defaults remain engine-local."""
            try:
                resolved = resolve_service_prompt(database, "media.web.summarization")
                return dict(resolved.parts) if resolved.source == "user" else {}
            finally:
                database.close_connection()

        parts = {**await asyncio.to_thread(load_saved), **parts}
    return MappingProxyType(parts)
