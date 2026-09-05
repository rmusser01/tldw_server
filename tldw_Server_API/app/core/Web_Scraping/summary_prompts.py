"""Request-bound overrides for synchronous web article summarization."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.Prompt_Management.service_prompts import resolve_service_prompt

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase


async def resolve_web_summary_overrides(
    payload: Any, get_prompts_db: Callable[[], Awaitable[PromptsDatabase]]
) -> Mapping[str, str] | None:
    """Freeze explicit/saved parts while leaving absent engine defaults untouched.

    Only the synchronous HTTP caller supplies owner-bound storage. No lookup is
    needed for disabled summarization or a complete pair of explicit parts.
    """
    if not payload.summarize_checkbox:
        return None
    parts = {
        key: value
        for key, value in (("system", payload.system_prompt), ("user", payload.custom_prompt))
        if value is not None
    }
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
