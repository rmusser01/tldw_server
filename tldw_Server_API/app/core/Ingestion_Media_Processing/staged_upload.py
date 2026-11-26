"""Manage pre-claim multipart staging under the API's configured temporary root."""

import errno
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import UUID

from loguru import logger


def upload_staging_directory(job_uuid: UUID) -> Path:
    """Derive the private upload directory solely from the server-issued Jobs UUID."""
    return Path(tempfile.gettempdir()) / f"media_ingest_job_{job_uuid}"


def cleanup_cancelled_upload(job: dict[str, Any], payload: dict[str, Any]) -> None:
    """Unlink the bound source and remove the directory only when it is empty."""
    if (
        job.get("domain") != "media_ingest"
        or job.get("job_type") != "media_ingest_item"
        or payload.get("cleanup_temp_dir") is not True
        or payload.get("source_kind") != "file"
        or not isinstance(payload.get("source"), str)
    ):
        return
    try:
        directory = upload_staging_directory(UUID(str(job.get("uuid"))))
        source = Path(payload["source"])
        if (
            payload.get("temp_dir") != str(directory)
            or source.parent != directory
            or source.name in {"", ".", ".."}
            or "\0" in source.name
        ):
            return
        if os.unlink in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW"):
            # Pin the real directory while unlinking, even if its pathname is replaced.
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with suppress(FileNotFoundError):
                    os.unlink(source.name, dir_fd=fd)
            finally:
                os.close(fd)
        else:
            # Windows also supports junctions, which Path.is_symlink alone misses.
            if directory.resolve() != directory.parent.resolve() / directory.name:
                return
            source.unlink(missing_ok=True)
        directory.rmdir()
    except (FileNotFoundError, ValueError):
        return
    except OSError as exc:
        if exc.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
            logger.warning("Could not release staged media upload for job {} (errno={})", job.get("id"), exc.errno)
