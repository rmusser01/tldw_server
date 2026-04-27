"""
Postgres content-backend startup validation extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any


def validate_startup_content_backend(*, logger: Any) -> None:
    try:
        _validate_postgres_content_backend()
        logger.info("App Startup: PostgreSQL content backend validated")
    except RuntimeError as exc:
        logger.exception(f"Startup aborted: {exc}")
        raise
    except ImportError as exc:
        logger.debug(f"Content backend validation skipped (import error): {exc}")


def _validate_postgres_content_backend() -> None:
    from tldw_Server_API.app.core.DB_Management.DB_Manager import (
        validate_postgres_content_backend,
    )

    validate_postgres_content_backend()
