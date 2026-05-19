"""
Startup PostgreSQL RLS helpers extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any, Callable


def run_pg_rls_auto_ensure(
    backend: Any,
    *,
    ensure_prompt_studio_rls: Callable[[Any], bool],
    ensure_chacha_rls: Callable[[Any], bool],
    logger_obj: Any,
) -> tuple[bool, bool]:
    """Apply both PostgreSQL RLS installers and log the combined result."""
    prompt_ok = ensure_prompt_studio_rls(backend)
    chacha_ok = ensure_chacha_rls(backend)
    logger_obj.info(
        "PG RLS ensure invoked (prompt_studio_applied={}, chacha_applied={})",
        prompt_ok,
        chacha_ok,
    )
    return prompt_ok, chacha_ok
