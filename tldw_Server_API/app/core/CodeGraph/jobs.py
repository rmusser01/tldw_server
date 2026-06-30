"""Core Jobs helpers for native CodeGraph indexing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.Jobs.manager import JobManager

from .config import CodeGraphSettings
from .models import WorkspaceResolution

CODEGRAPH_JOBS_DOMAIN = "codegraph"
CODEGRAPH_INDEX_JOB_TYPE = "codegraph_index"


def codegraph_jobs_queue() -> str:
    """Return the queue name used by CodeGraph index/sync jobs."""
    queue = (os.getenv("CODEGRAPH_JOBS_QUEUE") or "default").strip()
    return queue or "default"


def build_codegraph_index_job_payload(
    *,
    resolution: WorkspaceResolution,
    settings: CodeGraphSettings,
    operation: str,
    force: bool,
    languages: list[str] | None,
    max_files: int | None,
) -> dict[str, Any]:
    """Serialize a CodeGraph index/sync request into a JSON-safe Jobs payload."""
    normalized_operation = str(operation or "").strip().lower()
    if normalized_operation not in {"index", "sync"}:
        raise ValueError("operation must be index or sync")
    return {
        "operation": normalized_operation,
        "workspace_root": _resolved_path(resolution.workspace_root),
        "workspace_key": resolution.workspace_key,
        "workspace_id": resolution.workspace_id,
        "workspace_source": resolution.source,
        "index_db_path": _resolved_path(resolution.index_db_path),
        "settings": _settings_payload(settings),
        "force": bool(force),
        "languages": list(languages) if languages is not None else None,
        "max_files": int(max_files) if max_files is not None else None,
    }


def enqueue_codegraph_index_job(
    *,
    resolution: WorkspaceResolution,
    settings: CodeGraphSettings,
    operation: str,
    force: bool,
    languages: list[str] | None,
    max_files: int | None,
    owner_user_id: str | None,
    jm: JobManager | None = None,
) -> dict[str, Any]:
    """Create a core Jobs row for CodeGraph index or sync work."""
    manager = jm or JobManager()
    payload = build_codegraph_index_job_payload(
        resolution=resolution,
        settings=settings,
        operation=operation,
        force=force,
        languages=languages,
        max_files=max_files,
    )
    created = manager.create_job(
        domain=CODEGRAPH_JOBS_DOMAIN,
        queue=codegraph_jobs_queue(),
        job_type=CODEGRAPH_INDEX_JOB_TYPE,
        payload=payload,
        owner_user_id=owner_user_id,
        max_retries=0,
    )
    return manager.get_job(int(created["id"])) or created


def _settings_payload(settings: CodeGraphSettings) -> dict[str, Any]:
    """Return JSON-safe CodeGraph settings for a Jobs payload."""
    return {
        "index_base_dir": _resolved_path(settings.index_base_dir),
        "max_file_size_bytes": settings.max_file_size_bytes,
        "foreground_max_files": settings.foreground_max_files,
        "foreground_max_bytes": settings.foreground_max_bytes,
        "max_index_seconds": settings.max_index_seconds,
        "max_context_chars": settings.max_context_chars,
        "max_search_results": settings.max_search_results,
        "exclude_dirs": list(settings.exclude_dirs),
    }


def _resolved_path(path: Path) -> str:
    """Return a stable absolute path string for cross-process Jobs payloads."""
    return str(path.expanduser().resolve(strict=False))
