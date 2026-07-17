# app/core/Skills/skills_service.py
#
# Service for managing skills (CRUD operations and file management)
#
"""
Skills Service
==============

Manages skills stored as SKILL.md files in user directories:
- user_databases/{user_id}/skills/{skill_name}/SKILL.md

Provides:
- CRUD operations for skills
- Import/export functionality
- Context payload generation for LLM injection
"""

import asyncio
import contextlib
import os
import re
import shutil
import stat
import time
import zipfile
from collections.abc import AsyncIterator, Coroutine
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Optional, TypeVar

from loguru import logger

from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
    canonical_filesystem_digest,
)
from tldw_Server_API.app.core.Context_Integrity.resolver import (
    ContextIntegrityBlocked,
    ContextIntegrityResolver,
    get_global_context_integrity_resolver,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Infrastructure.distributed_lock import FileLock
from tldw_Server_API.app.core.Skills.exceptions import (
    SkillConflictError,
    SkillNotFoundError,
    SkillParseError,
    SkillsError,
    SkillStorageError,
    SkillValidationError,
)
from tldw_Server_API.app.core.Skills.runtime_metadata import build_skill_runtime_metadata
from tldw_Server_API.app.core.Skills.skill_parser import SkillFrontmatter, SkillParser

# Skill name validation pattern (same as in schemas)
SKILL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
# Supporting file name validation pattern (same as in schemas)
SUPPORTING_FILE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$")
MAX_SUPPORTING_FILES_COUNT = 20
MAX_SUPPORTING_FILE_BYTES = 500000
MAX_SUPPORTING_FILES_TOTAL_BYTES = 5 * 1024 * 1024  # 5MB
MAX_SKILL_MD_BYTES = 500000
MAX_ZIP_IMPORT_ENTRIES = 100
SKILLS_TRASH_LOCK_TIMEOUT_SECONDS = 10.0
SKILL_INTEGRITY_TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".py", ".sh"}
SkillFileFingerprint = tuple[tuple[str, int, int, int, int, int], ...]
_TrashResult = TypeVar("_TrashResult")
_TRACEBACK_ERROR_MARKERS = (
    "traceback (most recent call last):",
    "stack trace",
)


def _public_import_preview_error(message: str) -> str:
    """Return bounded validation text without traceback-shaped internals."""
    lines: list[str] = []
    for raw_line in str(message or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(marker in lowered for marker in _TRACEBACK_ERROR_MARKERS):
            continue
        if line.startswith('File "') and ", line " in line:
            continue
        lines.append(line)

    public_message = " ".join(lines).strip()
    return public_message[:300] if public_message else "Invalid skill import"


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        stat.S_IFMT(left.st_mode),
    ) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
    )


def _stat_mtime_ns(value: os.stat_result) -> int:
    return int(getattr(value, "st_mtime_ns", int(value.st_mtime * 1_000_000_000)))


def _fingerprint_entry(relative_path: str, value: os.stat_result) -> tuple[str, int, int, int, int, int]:
    return (
        relative_path,
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
        int(value.st_size),
        _stat_mtime_ns(value),
    )


def _fd_relative_walk_supported() -> bool:
    supports_dir_fd = getattr(os, "supports_dir_fd", set())
    return bool(hasattr(os, "fwalk") and os.open in supports_dir_fd and os.stat in supports_dir_fd)


def _read_regular_file_bytes_no_follow(path: Path) -> bytes:
    expected = path.lstat()
    if not stat.S_ISREG(expected.st_mode):
        raise OSError(f"Skill file is not a regular file: {path}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or not _same_file_identity(expected, opened):
            raise OSError(f"Skill file changed while being opened: {path}")
        with os.fdopen(fd, "rb", closefd=True) as file_obj:
            fd = -1
            return file_obj.read()
    finally:
        if fd >= 0:
            os.close(fd)


class SkillMetadata:
    """Metadata for a stored skill."""

    def __init__(
        self,
        id: str,
        name: str,
        description: Optional[str] = None,
        argument_hint: Optional[str] = None,
        disable_model_invocation: bool = False,
        user_invocable: bool = True,
        allowed_tools: Optional[list[str]] = None,
        model: Optional[str] = None,
        context: str = "inline",
        directory_path: str = "",
        content_hash: Optional[str] = None,
        created_at: Optional[datetime] = None,
        last_modified: Optional[datetime] = None,
        version: int = 1,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.argument_hint = argument_hint
        self.disable_model_invocation = disable_model_invocation
        self.user_invocable = user_invocable
        self.allowed_tools = allowed_tools or []
        self.model = model
        self.context = context
        self.directory_path = directory_path
        self.content_hash = content_hash
        self.created_at = created_at or datetime.now(timezone.utc)
        self.last_modified = last_modified or datetime.now(timezone.utc)
        self.version = version

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "argument_hint": self.argument_hint,
            "disable_model_invocation": self.disable_model_invocation,
            "user_invocable": self.user_invocable,
            "allowed_tools": self.allowed_tools,
            "model": self.model,
            "context": self.context,
            "directory_path": self.directory_path,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillMetadata":
        created_at = data.get("created_at")
        last_modified = data.get("last_modified")

        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if isinstance(last_modified, str):
            last_modified = datetime.fromisoformat(last_modified)

        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            argument_hint=data.get("argument_hint"),
            disable_model_invocation=data.get("disable_model_invocation", False),
            user_invocable=data.get("user_invocable", True),
            allowed_tools=data.get("allowed_tools"),
            model=data.get("model"),
            context=data.get("context", "inline"),
            directory_path=data.get("directory_path", ""),
            content_hash=data.get("content_hash"),
            created_at=created_at,
            last_modified=last_modified,
            version=data.get("version", 1),
        )


class SkillsService:
    """Central service for skill management."""

    def __init__(
        self,
        user_id: int,
        base_path: Path,
        db: CharactersRAGDB | None = None,
        sync_interval: float = 5.0,
        integrity_resolver: ContextIntegrityResolver | None = None,
    ):
        """
        Initialize the SkillsService.

        Args:
            user_id: The user ID for skill isolation
            base_path: Base path for user databases (e.g., Databases/user_databases/{user_id}/)
            db: CharactersRAGDB instance for skill registry persistence
            sync_interval: Minimum seconds between filesystem syncs for read operations
        """
        self.user_id = user_id
        self.base_path = Path(base_path)
        self.skills_dir = self.base_path / "skills"
        self.trash_dir = self.skills_dir / ".trash"
        self.cleanup_dir = self.trash_dir / ".cleanup"
        self.trash_lock_path = self.skills_dir / ".trash.lock"
        self.db = db
        self._parser = SkillParser()
        self._sync_interval = sync_interval
        self._last_sync_time: float | None = None
        self._startup_maintenance_complete = False
        self.integrity_resolver = (
            integrity_resolver if integrity_resolver is not None else get_global_context_integrity_resolver()
        )
        self._integrity_decision_cache: dict[tuple[str, str], tuple[float, SkillFileFingerprint, bool]] = {}
        self._ensure_skills_directory()
        self._ensure_registry_ready()

    def _ensure_skills_directory(self) -> None:
        """Ensure the skills directory exists."""
        try:
            self.skills_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise SkillStorageError(
                f"Failed to create skills directory: {e}",
                path=str(self.skills_dir),
            ) from e

    def _ensure_registry_ready(self) -> None:
        """Ensure the skill registry table is available."""
        if self.db is None:
            raise SkillsError("SkillsService requires a database instance for registry operations.")
        try:
            self.db._ensure_skill_registry_table()
        except CharactersRAGDBError as e:
            raise SkillsError(f"Failed to ensure skill registry table: {e}") from e

    def _get_db(self) -> CharactersRAGDB:
        if self.db is None:
            raise SkillsError("SkillsService requires a database instance for registry operations.")
        return self.db

    @staticmethod
    async def _await_task_completion(
        task: asyncio.Task[_TrashResult],
    ) -> tuple[_TrashResult, asyncio.CancelledError | None]:
        """Wait for a task to finish while recording cancellation of the caller."""
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                return await asyncio.shield(task), cancellation
            except asyncio.CancelledError as error:
                if task.cancelled():
                    raise
                if cancellation is None:
                    cancellation = error

    async def _finish_trash_mutation(
        self,
        mutation: Coroutine[Any, Any, _TrashResult],
    ) -> _TrashResult:
        """Finish a locked Trash transaction before propagating caller cancellation."""
        task = asyncio.create_task(mutation)
        result, cancellation = await self._await_task_completion(task)
        if cancellation is not None:
            raise cancellation
        return result

    @contextlib.asynccontextmanager
    async def _trash_operation_lock(self) -> AsyncIterator[None]:
        """Serialize Trash filesystem and registry transitions for this user."""
        lock = FileLock(
            self.trash_lock_path,
            timeout=SKILLS_TRASH_LOCK_TIMEOUT_SECONDS,
        )
        acquire_task = asyncio.create_task(asyncio.to_thread(lock.acquire))
        acquired, cancellation = await self._await_task_completion(acquire_task)
        if cancellation is not None:
            if acquired:
                lock.release()
            raise cancellation
        if not acquired:
            raise SkillStorageError(
                "Skills Trash is busy; try again.",
                path=str(self.trash_dir),
            )
        try:
            yield
        finally:
            lock.release()

    def _get_skill_dir(self, name: str) -> Path:
        """Get the directory path for a skill."""
        safe_name = self._normalize_and_validate_skill_name(name)
        base = self.skills_dir.resolve(strict=False)
        path = (base / safe_name).resolve(strict=False)
        try:
            path.relative_to(base)
        except ValueError as e:
            raise SkillValidationError(f"Invalid skill name: {name}", field="name") from e
        return path

    def _ensure_trash_directory(self) -> Path:
        """Return a non-symlinked Trash directory under the user skills root."""
        try:
            self.trash_dir.mkdir(parents=True, exist_ok=True)
            trash_stat = self.trash_dir.lstat()
        except OSError as e:
            raise SkillStorageError(
                f"Failed to prepare Skills Trash: {e}",
                path=str(self.trash_dir),
            ) from e
        if not stat.S_ISDIR(trash_stat.st_mode) or stat.S_ISLNK(trash_stat.st_mode):
            raise SkillStorageError(
                "Skills Trash must be a regular directory",
                path=str(self.trash_dir),
            )
        return self.trash_dir

    def _get_archive_dir(self, row: dict[str, Any]) -> Path:
        """Resolve the archive directory for a deleted registry row."""
        trash_dir = self._ensure_trash_directory()
        trash_root = trash_dir.resolve(strict=False)
        archive_id = str(row.get("uuid") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", archive_id):
            raise SkillStorageError(
                "Skill archive identifier is invalid",
                path=str(row.get("directory_path") or ""),
            )
        archive_path = trash_root / archive_id

        try:
            archive_path.relative_to(trash_root)
        except ValueError as e:
            raise SkillStorageError(
                "Skill archive path escapes Skills Trash",
                path=str(archive_path),
            ) from e
        try:
            archive_stat = archive_path.lstat()
        except FileNotFoundError:
            pass
        except OSError as e:
            raise SkillStorageError(
                f"Failed to inspect skill archive path: {e}",
                path=str(archive_path),
            ) from e
        else:
            if stat.S_ISLNK(archive_stat.st_mode):
                raise SkillStorageError(
                    "Skill archive path must not be a symlink",
                    path=str(archive_path),
                )
        return archive_path

    def _ensure_cleanup_directory(self) -> Path:
        """Return the non-symlinked directory used for committed cleanup work."""
        self._ensure_trash_directory()
        try:
            self.cleanup_dir.mkdir(parents=True, exist_ok=True)
            cleanup_stat = self.cleanup_dir.lstat()
        except OSError as e:
            raise SkillStorageError(
                f"Failed to prepare Skills cleanup directory: {e}",
                path=str(self.cleanup_dir),
            ) from e
        if not stat.S_ISDIR(cleanup_stat.st_mode) or stat.S_ISLNK(cleanup_stat.st_mode):
            raise SkillStorageError(
                "Skills cleanup path must be a regular directory",
                path=str(self.cleanup_dir),
            )
        return self.cleanup_dir

    def _stage_for_cleanup(self, source: Path, label: str) -> Path:
        """Atomically move a committed obsolete bundle into the cleanup queue."""
        cleanup_dir = self._ensure_cleanup_directory()
        safe_label = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-") or "skill"
        destination = cleanup_dir / f"{safe_label}-{time.time_ns()}"
        self._move_skill_dir(source, destination)
        return destination

    def _remove_cleanup_path_best_effort(self, cleanup_path: Path) -> bool:
        """Remove one queued directory, preserving partial residue for a later retry."""
        cleanup_root = self.cleanup_dir.resolve(strict=False)
        path = Path(cleanup_path)
        try:
            path_stat = path.lstat()
            parent = path.parent.resolve(strict=False)
        except OSError as e:
            logger.warning(
                "Skipped invalid Skills cleanup entry '{}' for user {} (error {})",
                path.name,
                self.user_id,
                type(e).__name__,
            )
            return False
        if (
            parent != cleanup_root
            or not stat.S_ISDIR(path_stat.st_mode)
            or stat.S_ISLNK(path_stat.st_mode)
        ):
            logger.warning(
                "Skipped unsafe Skills cleanup entry '{}' for user {}",
                path.name,
                self.user_id,
            )
            return False

        def _retry_readonly(
            func: Any,
            readonly_path: str,
            exc_info: tuple[type[BaseException], BaseException, Any],
        ) -> None:
            error = exc_info[1]
            readonly_stat = os.lstat(readonly_path)
            if not isinstance(error, PermissionError) or stat.S_ISLNK(readonly_stat.st_mode):
                raise error
            os.chmod(readonly_path, readonly_stat.st_mode | stat.S_IWUSR)
            func(readonly_path)

        try:
            shutil.rmtree(path, onerror=_retry_readonly)
        except OSError as e:
            logger.warning(
                "Skills cleanup will retry entry '{}' for user {} (error {})",
                path.name,
                self.user_id,
                type(e).__name__,
            )
            return False
        return True

    def _retry_staged_cleanup(self) -> None:
        """Best-effort retry of cleanup work committed by an earlier service call."""
        try:
            cleanup_stat = self.cleanup_dir.lstat()
        except FileNotFoundError:
            return
        except OSError as e:
            logger.warning(
                "Could not inspect Skills cleanup directory for user {} (error {})",
                self.user_id,
                type(e).__name__,
            )
            return
        if not stat.S_ISDIR(cleanup_stat.st_mode) or stat.S_ISLNK(cleanup_stat.st_mode):
            logger.warning("Skipped unsafe Skills cleanup directory for user {}", self.user_id)
            return
        try:
            cleanup_paths = sorted(self.cleanup_dir.iterdir(), key=lambda path: path.name)
        except OSError as e:
            logger.warning(
                "Could not list Skills cleanup directory for user {} (error {})",
                self.user_id,
                type(e).__name__,
            )
            return
        for cleanup_path in cleanup_paths:
            self._remove_cleanup_path_best_effort(cleanup_path)

    def _registry_row_for_archive_id(self, archive_id: str) -> dict[str, Any] | None:
        """Resolve a Trash staging identifier without trusting mutable path metadata."""
        row = self._get_db().get_skill_registry_by_uuid(archive_id)
        if row is not None:
            return row
        if SKILL_NAME_PATTERN.fullmatch(archive_id):
            legacy_row = self._get_db().get_skill_registry(archive_id, include_deleted=True)
            if legacy_row is not None and not legacy_row.get("uuid"):
                return legacy_row
        return None

    def _reconcile_orphaned_archives(self) -> None:
        """Recover interrupted pre-commit moves and queue committed stale archives."""
        try:
            trash_stat = self.trash_dir.lstat()
        except FileNotFoundError:
            return
        except OSError as e:
            logger.warning(
                "Could not inspect Skills Trash for user {} (error {})",
                self.user_id,
                type(e).__name__,
            )
            return
        if not stat.S_ISDIR(trash_stat.st_mode) or stat.S_ISLNK(trash_stat.st_mode):
            logger.warning("Skipped unsafe Skills Trash reconciliation for user {}", self.user_id)
            return
        try:
            candidates = sorted(self.trash_dir.iterdir(), key=lambda path: path.name)
        except OSError as e:
            logger.warning(
                "Could not list Skills Trash for user {} (error {})",
                self.user_id,
                type(e).__name__,
            )
            return

        for candidate in candidates:
            try:
                candidate_stat = candidate.lstat()
            except OSError:
                continue
            if not stat.S_ISDIR(candidate_stat.st_mode) or stat.S_ISLNK(candidate_stat.st_mode):
                continue

            if candidate.name.startswith(".purging-"):
                archive_id = candidate.name.removeprefix(".purging-")
                try:
                    row = self._registry_row_for_archive_id(archive_id)
                    if row is None or not row.get("deleted"):
                        self._stage_for_cleanup(candidate, f"orphaned-purge-{archive_id}")
                        continue

                    archive_dir = self._get_archive_dir(row)
                    if archive_dir.exists():
                        logger.warning(
                            "Left duplicate purge staging '{}' for user {} because its Trash archive exists",
                            archive_id,
                            self.user_id,
                        )
                        continue
                    self._move_skill_dir(candidate, archive_dir)
                    logger.info("Restored interrupted purge staging for skill '{}'", row.get("name"))
                except (CharactersRAGDBError, OSError, SkillStorageError) as e:
                    logger.warning(
                        "Skills purge reconciliation deferred for archive '{}' and user {} (error {})",
                        archive_id,
                        self.user_id,
                        type(e).__name__,
                    )
                continue

            if candidate.name.startswith(".") or not re.fullmatch(r"[A-Za-z0-9_-]+", candidate.name):
                continue
            try:
                row = self._get_db().get_skill_registry_by_uuid(candidate.name)
                if row is not None and not row.get("deleted"):
                    active_name = self._normalize_and_validate_skill_name(str(row.get("name") or ""))
                    active_dir = self.skills_dir / active_name
                    try:
                        active_dir.lstat()
                    except FileNotFoundError:
                        active_exists = False
                    except OSError as e:
                        logger.warning(
                            "Preserved Skills archive '{}' for skill '{}' and user {} (inspect error {})",
                            candidate.name,
                            active_name,
                            self.user_id,
                            type(e).__name__,
                        )
                        continue
                    else:
                        active_exists = True

                    if not active_exists:
                        if not self._is_skill_bundle_valid(active_name, candidate):
                            logger.warning(
                                "Preserved invalid Skills archive '{}' for skill '{}' and user {}",
                                candidate.name,
                                active_name,
                                self.user_id,
                            )
                            continue
                        self._move_skill_dir(candidate, active_dir)
                        logger.info("Restored interrupted delete for skill '{}'", active_name)
                        continue

                    if not self._is_skill_bundle_valid(active_name, active_dir):
                        logger.warning(
                            "Preserved Skills archive '{}' because skill '{}' is invalid for user {}",
                            candidate.name,
                            active_name,
                            self.user_id,
                        )
                        continue
                    self._stage_for_cleanup(candidate, f"orphaned-replacement-{candidate.name}")
            except (CharactersRAGDBError, OSError, SkillStorageError, SkillValidationError) as e:
                logger.warning(
                    "Skills replacement reconciliation deferred for archive '{}' and user {} (error {})",
                    candidate.name,
                    self.user_id,
                    type(e).__name__,
                )

    def _discard_stale_archive(self, archive_dir: Path, label: str) -> None:
        """Move an obsolete canonical archive aside before reusing its path."""
        if not archive_dir.exists():
            return
        cleanup_path = self._stage_for_cleanup(archive_dir, label)
        self._remove_cleanup_path_best_effort(cleanup_path)

    def _move_skill_dir(self, source: Path, destination: Path) -> None:
        """Atomically move one non-symlinked skill bundle within the skills root."""
        root = self.skills_dir.resolve(strict=False)
        source_path = source.resolve(strict=False)
        destination_path = destination.resolve(strict=False)
        try:
            source_path.relative_to(root)
            destination_path.relative_to(root)
        except ValueError as e:
            raise OSError("Skill bundle move escapes the user skills root") from e

        source_stat = source.lstat()
        if not stat.S_ISDIR(source_stat.st_mode) or stat.S_ISLNK(source_stat.st_mode):
            raise OSError(f"Skill bundle is not a regular directory: {source}")
        try:
            destination.lstat()
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Skill bundle destination already exists: {destination}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)

    def _new_skill_staging_path(self, name: str, operation: str) -> Path:
        """Return a unique hidden path for preparing a complete skill bundle."""
        safe_name = self._normalize_and_validate_skill_name(name)
        safe_operation = re.sub(r"[^A-Za-z0-9_-]+", "-", operation).strip("-") or "write"
        return self.skills_dir / f".staging-{safe_operation}-{safe_name}-{time.time_ns()}"

    def _prepare_skill_bundle(
        self,
        name: str,
        content: str,
        supporting_files: dict[str, Optional[str]],
        *,
        operation: str,
    ) -> tuple[Path, Any]:
        """Write and validate a complete bundle without publishing its active path."""
        staging_dir = self._new_skill_staging_path(name, operation)
        try:
            staging_dir.mkdir(parents=False, exist_ok=False)
            self._skill_main_file(staging_dir).write_text(content, encoding="utf-8")
            for filename, file_content in supporting_files.items():
                file_path = self._safe_supporting_path(staging_dir, filename)
                file_path.write_text(file_content or "", encoding="utf-8")
            parsed = self._parse_unchecked_skill_directory(name, staging_dir)
            self._validate_parsed_skill_name(name, parsed)
            return staging_dir, parsed
        except (OSError, SkillStorageError) as e:
            with contextlib.suppress(OSError, SkillStorageError):
                self._remove_skill_dir(staging_dir)
            raise SkillStorageError(
                f"Failed to prepare skill bundle: {e}",
                path=str(staging_dir),
            ) from e
        except SkillValidationError:
            with contextlib.suppress(OSError, SkillStorageError):
                self._remove_skill_dir(staging_dir)
            raise
        except Exception as e:
            with contextlib.suppress(OSError, SkillStorageError):
                self._remove_skill_dir(staging_dir)
            raise SkillValidationError(f"Invalid skill content: {e}") from e

    def _replacement_backup_path(self, row: dict[str, Any]) -> Path:
        """Return the recoverable backup path for an active replacement."""
        replacement_id = str(row.get("uuid") or row.get("name") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", replacement_id):
            raise SkillStorageError("Skill replacement identifier is invalid")
        try:
            replacement_version = int(row.get("version"))
        except (TypeError, ValueError) as e:
            raise SkillStorageError("Skill replacement version is invalid") from e
        if replacement_version < 1:
            raise SkillStorageError("Skill replacement version is invalid")
        return self.skills_dir / f".replacing-{replacement_id}.v{replacement_version}"

    @staticmethod
    def _replacement_backup_metadata(backup_dir: Path) -> tuple[str, Optional[int]]:
        """Return the registry identifier and pre-swap version encoded in a backup."""
        suffix = backup_dir.name.removeprefix(".replacing-")
        versioned = re.fullmatch(r"([A-Za-z0-9_-]+)\.v([1-9][0-9]*)", suffix)
        if versioned:
            return versioned.group(1), int(versioned.group(2))
        if re.fullmatch(r"[A-Za-z0-9_-]+", suffix):
            return suffix, None
        raise SkillStorageError("Skill replacement backup name is invalid")

    def _bundle_content_hash(self, name: str, bundle_dir: Path) -> str | None:
        """Return a validated bundle hash, or None when the bundle is invalid."""
        try:
            parsed = self._parse_unchecked_skill_directory(name, bundle_dir)
            self._validate_parsed_skill_name(name, parsed)
        except Exception as e:
            logger.warning(
                "Invalid replacement bundle preserved for skill '{}' and user {} (error {})",
                name,
                self.user_id,
                type(e).__name__,
            )
            return None
        return str(parsed.content_hash)

    def _discard_prepublication_staging(self) -> None:
        """Remove hidden bundles that were never published after an interrupted write."""
        try:
            candidates = sorted(self.skills_dir.iterdir(), key=lambda path: path.name)
        except OSError as e:
            logger.warning(
                "Could not inspect Skills staging for user {} (error {})",
                self.user_id,
                type(e).__name__,
            )
            return

        for candidate in candidates:
            if not candidate.name.startswith(".staging-"):
                continue
            try:
                candidate_stat = candidate.lstat()
                if not stat.S_ISDIR(candidate_stat.st_mode) or stat.S_ISLNK(candidate_stat.st_mode):
                    continue
                cleanup_path = self._stage_for_cleanup(
                    candidate,
                    f"interrupted-staging-{candidate.name.removeprefix('.staging-')}",
                )
                self._remove_cleanup_path_best_effort(cleanup_path)
            except (OSError, SkillStorageError) as e:
                logger.warning(
                    "Skills staging cleanup deferred for '{}' and user {} (error {})",
                    candidate.name,
                    self.user_id,
                    type(e).__name__,
                )

    def _roll_back_interrupted_replacement(
        self,
        *,
        active_dir: Path,
        backup_dir: Path,
        replacement_id: str,
    ) -> None:
        """Restore a replacement backup without losing the displaced active bundle."""
        displaced_dir = self._stage_for_cleanup(
            active_dir,
            f"rollback-candidate-{replacement_id}",
        )
        try:
            self._move_skill_dir(backup_dir, active_dir)
        except Exception:
            try:
                self._move_skill_dir(displaced_dir, active_dir)
            except Exception as restore_error:
                raise SkillStorageError(
                    "Failed to restore the original or displaced skill bundle",
                    path=str(active_dir),
                ) from restore_error
            raise
        self._remove_cleanup_path_best_effort(displaced_dir)

    def _reconcile_interrupted_active_replacements(self) -> bool:
        """Resolve active bundle swaps and report whether registry sync is safe."""
        try:
            candidates = sorted(self.skills_dir.iterdir(), key=lambda path: path.name)
        except OSError as e:
            logger.warning(
                "Could not inspect Skills replacements for user {} (error {})",
                self.user_id,
                type(e).__name__,
            )
            return False

        all_resolved = True
        for backup_dir in candidates:
            if not backup_dir.name.startswith(".replacing-"):
                continue
            replacement_label = backup_dir.name.removeprefix(".replacing-")
            try:
                replacement_id, replacement_version = self._replacement_backup_metadata(
                    backup_dir
                )
                backup_stat = backup_dir.lstat()
                if not stat.S_ISDIR(backup_stat.st_mode) or stat.S_ISLNK(backup_stat.st_mode):
                    all_resolved = False
                    logger.warning(
                        "Preserved unsafe Skills replacement '{}' for user {}",
                        replacement_id,
                        self.user_id,
                    )
                    continue
                row = self._registry_row_for_archive_id(replacement_id)
                if row is None or row.get("deleted"):
                    all_resolved = False
                    logger.warning(
                        "Preserved unresolved Skills replacement '{}' for user {}",
                        replacement_id,
                        self.user_id,
                    )
                    continue

                name = self._normalize_and_validate_skill_name(str(row.get("name") or ""))
                registry_hash = str(row.get("file_hash") or "")
                backup_hash = self._bundle_content_hash(name, backup_dir)
                if not registry_hash or backup_hash is None:
                    all_resolved = False
                    logger.warning(
                        "Preserved unverifiable Skills replacement '{}' for skill '{}' and user {}",
                        replacement_id,
                        name,
                        self.user_id,
                    )
                    continue
                try:
                    registry_version = int(row.get("version") or 1)
                except (TypeError, ValueError) as e:
                    raise SkillStorageError("Skill registry version is invalid") from e
                if registry_version < 1:
                    raise SkillStorageError("Skill registry version is invalid")

                active_dir = self._get_skill_dir(name)
                try:
                    active_stat = active_dir.lstat()
                except FileNotFoundError:
                    if (
                        backup_hash == registry_hash
                        and (
                            replacement_version is None
                            or registry_version == replacement_version
                        )
                    ):
                        self._move_skill_dir(backup_dir, active_dir)
                        logger.info("Restored interrupted replacement for skill '{}'", name)
                    else:
                        all_resolved = False
                        logger.warning(
                            "Preserved ambiguous Skills replacement '{}' for missing skill '{}' and user {}",
                            replacement_id,
                            name,
                            self.user_id,
                        )
                    continue

                if not stat.S_ISDIR(active_stat.st_mode) or stat.S_ISLNK(active_stat.st_mode):
                    all_resolved = False
                    logger.warning(
                        "Preserved Skills replacement '{}' because skill '{}' is unsafe for user {}",
                        replacement_id,
                        name,
                        self.user_id,
                    )
                    continue
                active_hash = self._bundle_content_hash(name, active_dir)
                if active_hash is None:
                    all_resolved = False
                    continue

                if replacement_version is not None:
                    if registry_version == replacement_version:
                        if backup_hash != registry_hash:
                            all_resolved = False
                            logger.warning(
                                "Preserved invalid Skills replacement backup '{}' for skill '{}' and user {}",
                                replacement_id,
                                name,
                                self.user_id,
                            )
                            continue
                        self._roll_back_interrupted_replacement(
                            active_dir=active_dir,
                            backup_dir=backup_dir,
                            replacement_id=replacement_id,
                        )
                        logger.info("Rolled back interrupted replacement for skill '{}'", name)
                        continue

                    if registry_version > replacement_version and active_hash == registry_hash:
                        cleanup_path = self._stage_for_cleanup(
                            backup_dir,
                            f"committed-replacement-{replacement_id}",
                        )
                        self._remove_cleanup_path_best_effort(cleanup_path)
                        continue

                    all_resolved = False
                    logger.warning(
                        "Preserved ambiguous Skills replacement '{}' for skill '{}' and user {}",
                        replacement_id,
                        name,
                        self.user_id,
                    )
                    continue

                if active_hash == registry_hash and backup_hash != registry_hash:
                    cleanup_path = self._stage_for_cleanup(
                        backup_dir,
                        f"committed-replacement-{replacement_id}",
                    )
                    self._remove_cleanup_path_best_effort(cleanup_path)
                    continue
                if backup_hash != registry_hash or active_hash == registry_hash:
                    all_resolved = False
                    logger.warning(
                        "Preserved ambiguous Skills replacement '{}' for skill '{}' and user {}",
                        replacement_id,
                        name,
                        self.user_id,
                    )
                    continue

                self._roll_back_interrupted_replacement(
                    active_dir=active_dir,
                    backup_dir=backup_dir,
                    replacement_id=replacement_id,
                )
                logger.info("Rolled back interrupted replacement for skill '{}'", name)
            except (CharactersRAGDBError, OSError, SkillStorageError, SkillValidationError) as e:
                all_resolved = False
                logger.warning(
                    "Skills replacement recovery deferred for '{}' and user {} (error {})",
                    replacement_label,
                    self.user_id,
                    type(e).__name__,
                )
        return all_resolved

    def _is_archive_restorable(self, archive_dir: Path) -> bool:
        """Return whether an archived bundle has a regular directory and SKILL.md."""
        try:
            archive_stat = archive_dir.lstat()
            skill_stat = (archive_dir / "SKILL.md").lstat()
        except OSError:
            return False
        return (
            stat.S_ISDIR(archive_stat.st_mode)
            and not stat.S_ISLNK(archive_stat.st_mode)
            and stat.S_ISREG(skill_stat.st_mode)
            and not stat.S_ISLNK(skill_stat.st_mode)
        )

    def _is_skill_bundle_valid(self, name: str, bundle_dir: Path) -> bool:
        """Return whether a bundle can be read safely and parsed as a skill."""
        if not self._is_archive_restorable(bundle_dir):
            return False
        try:
            parsed = self._parse_unchecked_skill_directory(name, bundle_dir)
            self._validate_parsed_skill_name(name, parsed)
        except Exception as e:
            logger.warning(
                "Invalid bundle preserved for skill '{}' and user {} (error {})",
                name,
                self.user_id,
                type(e).__name__,
            )
            return False
        return True

    def _remove_skill_dir(self, skill_dir: Path) -> None:
        """Remove a skill directory after confirming it is under the user skills root."""
        base = self.skills_dir.resolve(strict=False)
        path = skill_dir.resolve(strict=False)
        try:
            path.relative_to(base)
        except ValueError as e:
            raise SkillStorageError("Refusing to remove skill directory outside skills root", path=str(skill_dir)) from e
        shutil.rmtree(path, ignore_errors=True)

    def _skill_main_file(self, skill_dir: Path) -> Path:
        """Return the validated SKILL.md path for a skill directory."""
        base = skill_dir.resolve(strict=False)
        path = (base / "SKILL.md").resolve(strict=False)
        try:
            path.relative_to(base)
        except ValueError as e:
            raise SkillStorageError("SKILL.md path escapes skill directory", path=str(path)) from e
        return path

    def _skill_asset_id(self, name: str) -> str:
        """Return the Context Integrity asset id for a user skill."""
        return f"skill:user:{self.user_id}/{name}"

    def _relative_skill_path(self, skill_dir: Path, path: Path) -> str:
        try:
            return path.relative_to(skill_dir).as_posix()
        except ValueError:
            return path.resolve().relative_to(skill_dir.resolve()).as_posix()

    def _read_skill_file_map_fd_walk(self, skill_dir: Path) -> dict[str, bytes]:
        root_stat = skill_dir.lstat()
        if not stat.S_ISDIR(root_stat.st_mode):
            raise OSError(f"Skill directory is not a directory: {skill_dir}")

        files: dict[str, bytes] = {}
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        for directory, dirnames, filenames, dirfd in os.fwalk(skill_dir, topdown=True, follow_symlinks=False):
            dirnames.sort()
            for dirname in list(dirnames):
                entry_path = Path(directory) / dirname
                entry_stat = os.stat(dirname, dir_fd=dirfd, follow_symlinks=False)
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise OSError(f"Symlinked skill path is not allowed: {entry_path}")
                if not stat.S_ISDIR(entry_stat.st_mode):
                    dirnames.remove(dirname)

            for filename in sorted(filenames):
                entry_path = Path(directory) / filename
                entry_stat = os.stat(filename, dir_fd=dirfd, follow_symlinks=False)
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise OSError(f"Symlinked skill path is not allowed: {entry_path}")
                if not stat.S_ISREG(entry_stat.st_mode):
                    continue
                if Path(filename).suffix.lower() not in SKILL_INTEGRITY_TEXT_SUFFIXES:
                    continue

                fd = os.open(filename, flags, dir_fd=dirfd)
                try:
                    opened_stat = os.fstat(fd)
                    if not stat.S_ISREG(opened_stat.st_mode) or not _same_file_identity(entry_stat, opened_stat):
                        raise OSError(f"Skill file changed while being opened: {entry_path}")
                    with os.fdopen(fd, "rb", closefd=True) as file_obj:
                        fd = -1
                        files[self._relative_skill_path(skill_dir, entry_path)] = file_obj.read()
                finally:
                    if fd >= 0:
                        os.close(fd)
        return files

    def _read_skill_file_map_path_walk(self, skill_dir: Path) -> dict[str, bytes]:
        root_stat = skill_dir.lstat()
        if not stat.S_ISDIR(root_stat.st_mode):
            raise OSError(f"Skill directory is not a directory: {skill_dir}")

        files: dict[str, bytes] = {}

        def _walk(directory: Path, relative_prefix: str = "") -> None:
            for path in sorted(directory.iterdir(), key=lambda item: item.name):
                entry_stat = path.lstat()
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise OSError(f"Symlinked skill path is not allowed: {path}")

                relative_path = f"{relative_prefix}{path.name}"
                if stat.S_ISDIR(entry_stat.st_mode):
                    _walk(path, f"{relative_path}/")
                    continue

                if not stat.S_ISREG(entry_stat.st_mode):
                    continue
                if path.suffix.lower() not in SKILL_INTEGRITY_TEXT_SUFFIXES:
                    continue
                files[relative_path] = _read_regular_file_bytes_no_follow(path)

        _walk(skill_dir)
        return files

    def _read_skill_file_map(self, skill_dir: Path) -> dict[str, bytes]:
        """Read prompt-bearing skill files without following symlinks."""
        if _fd_relative_walk_supported():
            return self._read_skill_file_map_fd_walk(skill_dir)
        return self._read_skill_file_map_path_walk(skill_dir)

    def _skill_file_fingerprint_fd_walk(self, skill_dir: Path) -> SkillFileFingerprint:
        root_stat = skill_dir.lstat()
        if not stat.S_ISDIR(root_stat.st_mode):
            raise OSError(f"Skill directory is not a directory: {skill_dir}")

        entries: list[tuple[str, int, int, int, int, int]] = []
        for directory, dirnames, filenames, dirfd in os.fwalk(skill_dir, topdown=True, follow_symlinks=False):
            dirnames.sort()
            for dirname in list(dirnames):
                entry_path = Path(directory) / dirname
                entry_stat = os.stat(dirname, dir_fd=dirfd, follow_symlinks=False)
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise OSError(f"Symlinked skill path is not allowed: {entry_path}")
                if not stat.S_ISDIR(entry_stat.st_mode):
                    dirnames.remove(dirname)

            for filename in sorted(filenames):
                entry_path = Path(directory) / filename
                entry_stat = os.stat(filename, dir_fd=dirfd, follow_symlinks=False)
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise OSError(f"Symlinked skill path is not allowed: {entry_path}")
                if not stat.S_ISREG(entry_stat.st_mode):
                    continue
                if Path(filename).suffix.lower() not in SKILL_INTEGRITY_TEXT_SUFFIXES:
                    continue
                entries.append(_fingerprint_entry(self._relative_skill_path(skill_dir, entry_path), entry_stat))
        return tuple(entries)

    def _skill_file_fingerprint_path_walk(self, skill_dir: Path) -> SkillFileFingerprint:
        root_stat = skill_dir.lstat()
        if not stat.S_ISDIR(root_stat.st_mode):
            raise OSError(f"Skill directory is not a directory: {skill_dir}")

        entries: list[tuple[str, int, int, int, int, int]] = []

        def _walk(directory: Path, relative_prefix: str = "") -> None:
            for path in sorted(directory.iterdir(), key=lambda item: item.name):
                entry_stat = path.lstat()
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise OSError(f"Symlinked skill path is not allowed: {path}")

                relative_path = f"{relative_prefix}{path.name}"
                if stat.S_ISDIR(entry_stat.st_mode):
                    _walk(path, f"{relative_path}/")
                    continue

                if not stat.S_ISREG(entry_stat.st_mode):
                    continue
                if path.suffix.lower() not in SKILL_INTEGRITY_TEXT_SUFFIXES:
                    continue
                entries.append(_fingerprint_entry(relative_path, entry_stat))

        _walk(skill_dir)
        return tuple(entries)

    def _skill_file_fingerprint(self, skill_dir: Path) -> SkillFileFingerprint:
        if _fd_relative_walk_supported():
            return self._skill_file_fingerprint_fd_walk(skill_dir)
        return self._skill_file_fingerprint_path_walk(skill_dir)

    def _skill_digest(self, name: str, files: dict[str, bytes]) -> str:
        """Compute the same canonical digest used by startup skill inventory."""
        return canonical_filesystem_digest(
            source_type="skill_file",
            asset_id=self._skill_asset_id(name),
            files=files,
            metadata={"skill_name": name},
        )

    def _is_skill_allowed(self, name: str, *, purpose: str) -> bool:
        """Return whether a skill can be advertised for a model-facing purpose."""
        if self.integrity_resolver is None:
            return True

        cache_key = (name, purpose)
        now = time.monotonic()
        try:
            skill_dir = self._get_skill_dir(name)
            fingerprint = self._skill_file_fingerprint(skill_dir)
            cached = self._integrity_decision_cache.get(cache_key)
            if cached is not None and cached[0] >= now and cached[1] == fingerprint:
                return cached[2]

            files = self._read_skill_file_map(skill_dir)
            if "SKILL.md" not in files:
                self._integrity_decision_cache[cache_key] = (
                    now + min(max(self._sync_interval, 0.0), 5.0),
                    fingerprint,
                    False,
                )
                return False
            current_digest = self._skill_digest(name, files)
            self.integrity_resolver.require_digest_allowed(
                self._skill_asset_id(name),
                current_digest=current_digest,
                purpose=purpose,
            )
            self._integrity_decision_cache[cache_key] = (
                now + min(max(self._sync_interval, 0.0), 5.0),
                fingerprint,
                True,
            )
            return True
        except (ContextIntegrityBlocked, OSError, UnicodeDecodeError):
            self._integrity_decision_cache.pop(cache_key, None)
            return False

    def _require_skill_allowed(
        self,
        name: str,
        *,
        purpose: str,
        current_digest: str | None = None,
    ) -> None:
        """Raise when a skill is not allowed by the current integrity resolver."""
        if self.integrity_resolver is None:
            return

        asset_id = self._skill_asset_id(name)
        if current_digest is None:
            self.integrity_resolver.require_allowed(asset_id, purpose=purpose)
            return

        self.integrity_resolver.require_digest_allowed(
            asset_id,
            current_digest=current_digest,
            purpose=purpose,
        )

    def _parse_unchecked_skill_directory(
        self,
        name: str,
        skill_dir: Path,
        *,
        files: dict[str, bytes] | None = None,
    ) -> Any:
        """Parse a skill from an already-read file map without integrity checks."""
        files = files if files is not None else self._read_skill_file_map(skill_dir)
        raw_skill = files.get("SKILL.md")
        if raw_skill is None:
            raise SkillNotFoundError(name, detail="SKILL.md not found")

        parsed = self._parser.parse_content(raw_skill.decode("utf-8"), default_name=name)
        parsed.supporting_files = {
            relative_path: content.decode("utf-8")
            for relative_path, content in files.items()
            if relative_path != "SKILL.md"
        }
        return parsed

    def _parse_verified_skill_directory(self, name: str, skill_dir: Path) -> Any:
        """Read, verify, and parse a skill using one file snapshot."""
        files = self._read_skill_file_map(skill_dir)
        current_digest = self._skill_digest(name, files)
        self._require_skill_allowed(
            name,
            purpose="skill_read",
            current_digest=current_digest,
        )
        return self._parse_unchecked_skill_directory(name, skill_dir, files=files)

    def _normalize_and_validate_skill_name(self, name: str, *, source: str = "skill name") -> str:
        """Normalize and validate a skill name."""
        normalized = (name or "").strip().lower()
        if not normalized:
            raise SkillValidationError(f"{source.capitalize()} must be specified", field="name")
        if not SKILL_NAME_PATTERN.match(normalized):
            raise SkillValidationError(
                f"Invalid {source}: '{name}'. "
                "Skill name must start with a letter and contain only lowercase letters, numbers, and hyphens.",
                field="name",
            )
        return normalized

    def _resolve_import_skill_name(
        self,
        parsed: Any,
        name: Optional[str],
    ) -> tuple[str, Optional[str]]:
        """Resolve one canonical import name with shared preview/import validation."""
        frontmatter_name = parsed.frontmatter.name
        if frontmatter_name is not None and not isinstance(frontmatter_name, str):
            raise SkillValidationError(
                "Frontmatter skill name must be a string",
                field="name",
            )
        normalized_frontmatter_name = (
            self._normalize_and_validate_skill_name(
                frontmatter_name,
                source="frontmatter skill name",
            )
            if frontmatter_name
            else None
        )
        requested_name = (
            self._normalize_and_validate_skill_name(name)
            if name is not None
            else None
        )
        skill_name = requested_name or normalized_frontmatter_name
        if not skill_name:
            raise SkillValidationError(
                "Skill name must be specified in frontmatter or as parameter"
            )
        return skill_name, requested_name

    def _validate_parsed_skill_name(self, canonical_name: str, parsed: Any) -> None:
        """Require parsed frontmatter identity to match the canonical registry name."""
        if not isinstance(parsed.frontmatter.name, str):
            raise SkillValidationError(
                "Frontmatter skill name must be a string",
                field="name",
            )
        parsed_name = self._normalize_and_validate_skill_name(
            parsed.frontmatter.name,
            source="frontmatter skill name",
        )
        if parsed_name != canonical_name:
            raise SkillValidationError(
                f"Frontmatter skill name '{parsed_name}' must match canonical name "
                f"'{canonical_name}'",
                field="name",
            )

    def _validate_supporting_filename(self, filename: str) -> str:
        """Validate a supporting filename and return normalized value."""
        normalized = (filename or "").strip()
        if not normalized:
            raise SkillValidationError("Supporting file name cannot be empty", field="supporting_files")
        if normalized.lower() == "skill.md":
            raise SkillValidationError("SKILL.md cannot be a supporting file", field="supporting_files")
        if not SUPPORTING_FILE_NAME_PATTERN.match(normalized):
            raise SkillValidationError(f"Invalid supporting file name: {filename}", field="supporting_files")
        return normalized

    def _safe_supporting_path(self, skill_dir: Path, filename: str) -> Path:
        """Build a validated supporting-file path constrained to the skill directory."""
        path = (skill_dir / filename).resolve()
        base = skill_dir.resolve()
        try:
            path.relative_to(base)
        except ValueError as e:
            raise SkillValidationError(
                f"Supporting file path escapes skill directory: {filename}",
                field="supporting_files",
            ) from e
        return path

    def _normalize_supporting_files(
        self,
        supporting_files: Optional[dict[str, Any]],
        *,
        allow_deletes: bool,
    ) -> dict[str, Optional[str]]:
        """Validate and normalize supporting files payload."""
        if not supporting_files:
            return {}

        normalized: dict[str, Optional[str]] = {}
        non_null_count = 0
        total_bytes = 0
        for raw_name, raw_content in supporting_files.items():
            if not isinstance(raw_name, str):
                raise SkillValidationError("Supporting file names must be strings", field="supporting_files")
            filename = self._validate_supporting_filename(raw_name)

            if raw_content is None:
                if allow_deletes:
                    normalized[filename] = None
                    continue
                raise SkillValidationError(
                    f"Supporting file '{filename}' content cannot be null",
                    field="supporting_files",
                )

            if not isinstance(raw_content, str):
                raise SkillValidationError(
                    f"Supporting file '{filename}' content must be a string",
                    field="supporting_files",
                )
            non_null_count += 1
            if non_null_count > MAX_SUPPORTING_FILES_COUNT:
                raise SkillValidationError(
                    f"Too many supporting files ({non_null_count}); maximum is {MAX_SUPPORTING_FILES_COUNT}",
                    field="supporting_files",
                )

            file_bytes = len(raw_content.encode("utf-8"))
            if file_bytes > MAX_SUPPORTING_FILE_BYTES:
                raise SkillValidationError(
                    f"Supporting file {filename} exceeds 500KB limit",
                    field="supporting_files",
                )
            total_bytes += file_bytes
            if total_bytes > MAX_SUPPORTING_FILES_TOTAL_BYTES:
                raise SkillValidationError(
                    f"Total supporting files size ({total_bytes} bytes) exceeds "
                    f"{MAX_SUPPORTING_FILES_TOTAL_BYTES // (1024 * 1024)}MB limit",
                    field="supporting_files",
                )
            normalized[filename] = raw_content

        return normalized

    def _parse_skill_file(self, skill_dir: Path) -> Optional[Any]:
        """Parse SKILL.md content without loading supporting files."""
        skill_file = self._skill_main_file(skill_dir)
        try:
            content = _read_regular_file_bytes_no_follow(skill_file).decode("utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to read SKILL.md for {skill_dir.name}: {e}")
            return None
        try:
            return self._parser.parse_content(content, default_name=skill_dir.name)
        except Exception as e:
            logger.warning(f"Failed to parse SKILL.md for {skill_dir.name}: {e}")
            return None

    @staticmethod
    def _registry_payload(name: str, skill_dir: Path, parsed: Any) -> dict[str, Any]:
        """Build registry metadata from parsed skill content."""
        return {
            "name": name,
            "description": parsed.frontmatter.description,
            "argument_hint": parsed.frontmatter.argument_hint,
            "disable_model_invocation": parsed.frontmatter.disable_model_invocation,
            "user_invocable": parsed.frontmatter.user_invocable,
            "allowed_tools": parsed.frontmatter.allowed_tools,
            "model": parsed.frontmatter.model,
            "context": parsed.frontmatter.context,
            "directory_path": str(skill_dir),
            "file_hash": parsed.content_hash,
        }

    def _replace_active_bundle_locked(
        self,
        name: str,
        staging_dir: Path,
        parsed: Any,
        active_row: dict[str, Any],
    ) -> None:
        """Atomically publish an active replacement with startup-recoverable rollback."""
        active_dir = self._get_skill_dir(name)
        backup_dir = self._replacement_backup_path(active_row)
        if backup_dir.exists():
            raise SkillStorageError(
                "A previous skill replacement still requires recovery",
                path=str(backup_dir),
            )

        try:
            self._move_skill_dir(active_dir, backup_dir)
            try:
                self._move_skill_dir(staging_dir, active_dir)
            except Exception:
                self._move_skill_dir(backup_dir, active_dir)
                raise

            try:
                self._get_db().update_skill_registry(
                    name,
                    self._registry_payload(name, active_dir, parsed),
                    expected_version=int(active_row.get("version") or 1),
                )
            except Exception:
                try:
                    self._move_skill_dir(active_dir, staging_dir)
                    self._move_skill_dir(backup_dir, active_dir)
                except (OSError, SkillStorageError) as rollback_error:
                    raise SkillsError(
                        f"Failed to replace skill '{name}' and restore the original"
                    ) from rollback_error
                finally:
                    with contextlib.suppress(OSError, SkillStorageError):
                        self._remove_skill_dir(staging_dir)
                raise
        except FileNotFoundError as e:
            with contextlib.suppress(OSError, SkillStorageError):
                self._remove_skill_dir(staging_dir)
            raise SkillStorageError(
                f"Active skill bundle for '{name}' was not found",
                path=str(active_dir),
            ) from e
        except OSError as e:
            with contextlib.suppress(OSError, SkillStorageError):
                self._remove_skill_dir(staging_dir)
            raise SkillStorageError(
                f"Failed to publish replacement for skill '{name}': {e}",
                path=str(active_dir),
            ) from e

        try:
            cleanup_path = self._stage_for_cleanup(
                backup_dir,
                f"committed-replacement-{active_row.get('uuid') or name}",
            )
        except (OSError, SkillStorageError) as e:
            logger.warning(
                "Replacement for '{}' committed for user {}; backup cleanup deferred (error {})",
                name,
                self.user_id,
                type(e).__name__,
            )
            return
        self._remove_cleanup_path_best_effort(cleanup_path)

    def _activate_deleted_replacement_locked(
        self,
        name: str,
        skill_dir: Path,
        parsed: Any,
        deleted_row: dict[str, Any],
    ) -> None:
        """Activate a prepared replacement while the Trash lock is held."""
        deleted_version = int(deleted_row.get("version") or 1)
        archive_dir = self._get_archive_dir(deleted_row)
        try:
            self._get_db().restore_skill_registry(
                name,
                self._registry_payload(name, skill_dir, parsed),
                expected_version=deleted_version,
            )
        except Exception:
            self._remove_skill_dir(skill_dir)
            raise

        if not archive_dir.exists():
            return
        try:
            cleanup_path = self._stage_for_cleanup(archive_dir, f"replaced-{deleted_row.get('uuid') or name}")
        except (OSError, SkillStorageError) as e:
            logger.warning(
                "Replacement for '{}' is active for user {}; Trash cleanup deferred (error {})",
                name,
                self.user_id,
                type(e).__name__,
            )
            return
        self._remove_cleanup_path_best_effort(cleanup_path)

    def _metadata_from_row(self, row: dict[str, Any]) -> SkillMetadata:
        created_at = row.get("created_at")
        last_modified = row.get("last_modified")
        disable_model_invocation = row.get("disable_model_invocation")
        user_invocable = row.get("user_invocable")
        return SkillMetadata(
            id=row.get("uuid") or row.get("id"),
            name=row.get("name") or "",
            description=row.get("description"),
            argument_hint=row.get("argument_hint"),
            disable_model_invocation=(
                False if disable_model_invocation is None else bool(disable_model_invocation)
            ),
            user_invocable=True if user_invocable is None else bool(user_invocable),
            allowed_tools=row.get("allowed_tools"),
            model=row.get("model"),
            context=row.get("context", "inline"),
            directory_path=row.get("directory_path", ""),
            content_hash=row.get("file_hash"),
            created_at=created_at if isinstance(created_at, datetime) else datetime.now(timezone.utc),
            last_modified=last_modified if isinstance(last_modified, datetime) else datetime.now(timezone.utc),
            version=int(row.get("version") or 1),
        )

    def _sync_registry(self, force: bool = False) -> None:
        """Synchronize the registry while excluding concurrent Trash mutations."""
        lock = FileLock(
            self.trash_lock_path,
            timeout=SKILLS_TRASH_LOCK_TIMEOUT_SECONDS,
        )
        if not lock.acquire():
            raise SkillStorageError(
                "Skills Trash is busy; try again.",
                path=str(self.trash_dir),
            )
        try:
            self._sync_registry_locked(force=force)
        finally:
            lock.release()

    def _sync_registry_locked(self, force: bool = False) -> None:
        """Synchronize skill_registry while the per-user Trash lock is held.

        Args:
            force: If True, skip debounce and always sync. Write operations
                   should pass force=True.
        """
        if not self._reconcile_interrupted_active_replacements():
            raise SkillStorageError(
                "Skills replacement recovery is incomplete; retry after resolving preserved bundles",
                path=str(self.skills_dir),
            )
        if not self._startup_maintenance_complete:
            self._discard_prepublication_staging()
            self._reconcile_orphaned_archives()
            self._retry_staged_cleanup()
            self._startup_maintenance_complete = True

        now = time.monotonic()
        if (
            not force
            and self._last_sync_time is not None
            and (now - self._last_sync_time) < self._sync_interval
        ):
            return
        self._last_sync_time = now
        db = self._get_db()
        try:
            registry_rows = db.list_skill_registry(
                include_hidden=True,
                include_deleted=True,
                limit=10000,
                offset=0,
            )
        except CharactersRAGDBError as e:
            logger.error(f"Failed to read skill registry: {e}")
            raise SkillsError(f"Failed to read skill registry: {e}") from e

        registry_by_name = {row.get("name"): row for row in registry_rows if row and row.get("name")}

        disk_names: set[str] = set()
        if self.skills_dir.exists():
            for item in self.skills_dir.iterdir():
                if item.name.startswith("."):
                    continue
                try:
                    item_stat = item.lstat()
                except OSError as e:
                    logger.warning("Skipping unreadable skill path '{}': {}", item, e)
                    continue
                if stat.S_ISLNK(item_stat.st_mode):
                    logger.warning("Skipping symlinked skill directory '{}'", item)
                    continue
                if not stat.S_ISDIR(item_stat.st_mode):
                    continue

                disk_names.add(item.name)

                skill_file = item / "SKILL.md"
                try:
                    skill_file_stat = skill_file.lstat()
                except FileNotFoundError:
                    continue
                except OSError as e:
                    logger.warning("Skipping unreadable SKILL.md for '{}': {}", item.name, e)
                    continue
                if stat.S_ISLNK(skill_file_stat.st_mode):
                    logger.warning("Skipping skill '{}' because SKILL.md is a symlink", item.name)
                    continue
                if not stat.S_ISREG(skill_file_stat.st_mode):
                    continue

                parsed = self._parse_skill_file(item)
                if not parsed:
                    continue

                existing = registry_by_name.get(item.name)
                if existing is None:
                    try:
                        db.insert_skill_registry(
                            {
                                "name": item.name,
                                "description": parsed.frontmatter.description,
                                "argument_hint": parsed.frontmatter.argument_hint,
                                "disable_model_invocation": parsed.frontmatter.disable_model_invocation,
                                "user_invocable": parsed.frontmatter.user_invocable,
                                "allowed_tools": parsed.frontmatter.allowed_tools,
                                "model": parsed.frontmatter.model,
                                "context": parsed.frontmatter.context,
                                "directory_path": str(item),
                                "file_hash": parsed.content_hash,
                            }
                        )
                        logger.info(f"Indexed new skill '{item.name}' from disk")
                    except ConflictError:
                        logger.warning(f"Skill '{item.name}' already exists while syncing; skipping insert")
                    except CharactersRAGDBError as e:
                        logger.warning(f"Failed to insert skill '{item.name}' into registry: {e}")
                    continue

                if existing.get("deleted"):
                    update_data = {
                        "description": parsed.frontmatter.description,
                        "argument_hint": parsed.frontmatter.argument_hint,
                        "disable_model_invocation": parsed.frontmatter.disable_model_invocation,
                        "user_invocable": parsed.frontmatter.user_invocable,
                        "allowed_tools": parsed.frontmatter.allowed_tools,
                        "model": parsed.frontmatter.model,
                        "context": parsed.frontmatter.context,
                        "directory_path": str(item),
                        "file_hash": parsed.content_hash,
                    }
                    try:
                        db.restore_skill_registry(item.name, update_data, expected_version=existing.get("version", 1))
                        logger.info(f"Restored deleted skill '{item.name}' from disk")
                    except ConflictError as e:
                        logger.warning(f"Conflict restoring skill '{item.name}': {e}")
                    except CharactersRAGDBError as e:
                        logger.warning(f"Failed to restore skill '{item.name}': {e}")
                    continue

                if existing.get("file_hash") != parsed.content_hash:
                    update_data = {
                        "description": parsed.frontmatter.description,
                        "argument_hint": parsed.frontmatter.argument_hint,
                        "disable_model_invocation": parsed.frontmatter.disable_model_invocation,
                        "user_invocable": parsed.frontmatter.user_invocable,
                        "allowed_tools": parsed.frontmatter.allowed_tools,
                        "model": parsed.frontmatter.model,
                        "context": parsed.frontmatter.context,
                        "directory_path": str(item),
                        "file_hash": parsed.content_hash,
                    }
                    try:
                        db.update_skill_registry(item.name, update_data, expected_version=existing.get("version", 1))
                        logger.info(f"Updated skill registry for '{item.name}'")
                    except ConflictError as e:
                        logger.warning(f"Conflict updating skill '{item.name}': {e}")
                    except CharactersRAGDBError as e:
                        logger.warning(f"Failed to update skill '{item.name}': {e}")

        for name, row in registry_by_name.items():
            if name not in disk_names and not row.get("deleted"):
                try:
                    db.mark_skill_registry_deleted(name, expected_version=row.get("version", 1))
                    logger.info(f"Marked missing skill '{name}' as deleted")
                except ConflictError as e:
                    logger.warning(f"Conflict marking skill '{name}' deleted: {e}")
                except CharactersRAGDBError as e:
                    logger.warning(f"Failed to mark skill '{name}' deleted: {e}")

    async def _sync_registry_async(
        self,
        force: bool = False,
        *,
        trash_lock_held: bool = False,
    ) -> None:
        """Async wrapper for _sync_registry that offloads filesystem I/O to a thread."""
        sync = self._sync_registry_locked if trash_lock_held else self._sync_registry
        await asyncio.to_thread(sync, force=force)

    async def list_skills(
        self,
        include_hidden: bool = False,
        q: str | None = None,
        context: str | None = None,
        user_invocable: bool | None = None,
        has_tools: bool | None = None,
        model: str | None = None,
        sort: str = "name",
        order: str = "asc",
        limit: int = 100,
        offset: int = 0,
    ) -> list[SkillMetadata]:
        """
        List all skills for the user.

        Args:
            include_hidden: If True, include skills with user_invocable=False
            q: Optional case-insensitive search query for skill name or description
            context: Optional execution context filter ("inline" or "fork").
            user_invocable: Optional explicit visibility filter.
            has_tools: Optional filter for skills with non-empty allowed_tools.
            model: Optional exact model override filter.
            sort: Whitelisted sort field.
            order: Sort direction, "asc" or "desc".
            limit: Maximum number of skills to return
            offset: Offset for pagination

        Returns:
            List of skill metadata
        """
        await self._sync_registry_async()
        db = self._get_db()
        if self.integrity_resolver is not None:
            rows = db.list_skill_registry(
                include_hidden=include_hidden,
                include_deleted=False,
                q=q,
                context=context,
                user_invocable=user_invocable,
                has_tools=has_tools,
                model=model,
                sort=sort,
                order=order,
                limit=None,
                offset=0,
            )
            allowed_rows = [
                row for row in rows if self._is_skill_allowed(str(row.get("name") or ""), purpose="skill_discovery")
            ]
            return [self._metadata_from_row(row) for row in allowed_rows[offset : offset + limit]]

        rows = db.list_skill_registry(
            include_hidden=include_hidden,
            include_deleted=False,
            q=q,
            context=context,
            user_invocable=user_invocable,
            has_tools=has_tools,
            model=model,
            sort=sort,
            order=order,
            limit=limit,
            offset=offset,
        )
        return [self._metadata_from_row(row) for row in rows]

    def _is_model_visible_registry_row(self, row: dict[str, Any]) -> bool:
        """Return whether a registry row is eligible for model-facing discovery."""
        name = str(row.get("name") or "")
        user_invocable = row.get("user_invocable")
        disable_model_invocation = row.get("disable_model_invocation")
        return (
            (True if user_invocable is None else bool(user_invocable))
            and not (False if disable_model_invocation is None else bool(disable_model_invocation))
            and bool(name)
            and self._is_skill_allowed(name, purpose="skill_discovery")
        )

    def _list_model_visible_skills_page_sync(
        self,
        q: str | None,
        limit: int,
        offset: int,
    ) -> tuple[list[SkillMetadata], int]:
        """Return a filtered model-visible page and total from one registry query."""
        rows = self._get_db().list_skill_registry(
            include_hidden=True,
            include_deleted=False,
            q=q,
            sort="name",
            order="asc",
            limit=None,
            offset=0,
        )
        page: list[SkillMetadata] = []
        total = 0
        page_end = offset + limit
        for row in rows:
            if not self._is_model_visible_registry_row(row):
                continue
            if offset <= total < page_end:
                page.append(self._metadata_from_row(row))
            total += 1
        return page, total

    async def list_model_visible_skills_page(
        self,
        *,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[SkillMetadata], int]:
        """Return a model-visible Skill metadata page with its filtered total."""
        await self._sync_registry_async()
        return await asyncio.to_thread(
            self._list_model_visible_skills_page_sync,
            q,
            limit,
            offset,
        )

    def _get_model_visible_skill_metadata_sync(self, name: str) -> SkillMetadata:
        """Return metadata for one model-visible Skill or hide it as not found."""
        row = self._get_db().get_skill_registry(name, include_deleted=False)
        if not row or not self._is_model_visible_registry_row(row):
            raise SkillNotFoundError(name)
        return self._metadata_from_row(row)

    async def get_model_visible_skill_metadata(self, name: str) -> SkillMetadata:
        """Return metadata for an exact model-visible Skill lookup."""
        normalized = self._normalize_and_validate_skill_name(name)
        await self._sync_registry_async()
        return await asyncio.to_thread(
            self._get_model_visible_skill_metadata_sync,
            normalized,
        )

    def _get_skill_sync(self, name: str, *, enforce_integrity: bool) -> dict[str, Any]:
        """Load and verify a Skill after its asynchronous registry synchronization."""
        db = self._get_db()

        row = db.get_skill_registry(name, include_deleted=False)
        if not row:
            raise SkillNotFoundError(name)

        metadata = self._metadata_from_row(row)
        skill_dir = self._get_skill_dir(name)
        if not skill_dir.exists():
            with contextlib.suppress(Exception):
                db.mark_skill_registry_deleted(name, expected_version=metadata.version)
            raise SkillNotFoundError(name, detail="Skill directory not found")

        try:
            if enforce_integrity:
                parsed = self._parse_verified_skill_directory(name, skill_dir)
            else:
                parsed = self._parse_unchecked_skill_directory(name, skill_dir)
        except ContextIntegrityBlocked:
            raise
        except Exception as e:
            raise SkillsError(f"Failed to parse skill: {e}") from e

        return {
            "id": metadata.id,
            "name": metadata.name,
            "description": parsed.frontmatter.description,
            "argument_hint": parsed.frontmatter.argument_hint,
            "disable_model_invocation": parsed.frontmatter.disable_model_invocation,
            "user_invocable": parsed.frontmatter.user_invocable,
            "allowed_tools": parsed.frontmatter.allowed_tools,
            "model": parsed.frontmatter.model,
            "context": parsed.frontmatter.context,
            "content": parsed.content,
            "raw_content": parsed.raw_content,
            "supporting_files": parsed.supporting_files,
            "directory_path": str(skill_dir),
            "created_at": metadata.created_at,
            "last_modified": metadata.last_modified,
            "version": metadata.version,
        }

    async def get_skill(self, name: str, *, enforce_integrity: bool = True) -> dict[str, Any]:
        """
        Get full skill content.

        Args:
            name: The skill name

        Returns:
            Full skill data including content

        Raises:
            SkillNotFoundError: If skill doesn't exist
        """
        name = name.strip().lower()
        await self._sync_registry_async()
        return await asyncio.to_thread(
            self._get_skill_sync,
            name,
            enforce_integrity=enforce_integrity,
        )

    async def create_skill(
        self,
        name: str,
        content: str,
        supporting_files: Optional[dict[str, str]] = None,
        *,
        replace_deleted: bool = False,
    ) -> dict[str, Any]:
        """Create a skill while serializing filesystem and registry changes."""
        async with self._trash_operation_lock():
            return await self._finish_trash_mutation(
                self._create_skill_locked(
                    name,
                    content,
                    supporting_files,
                    replace_deleted=replace_deleted,
                )
            )

    async def _create_skill_locked(
        self,
        name: str,
        content: str,
        supporting_files: Optional[dict[str, str]] = None,
        *,
        replace_deleted: bool = False,
    ) -> dict[str, Any]:
        """
        Create a new skill while the per-user operation lock is held.

        Args:
            name: The skill name (lowercase, hyphens only)
            content: Full SKILL.md content with optional frontmatter
            supporting_files: Additional files to include
            replace_deleted: Replace a same-name Trash item after explicit confirmation

        Returns:
            Created skill data

        Raises:
            SkillConflictError: If skill with this name already exists
            SkillValidationError: If content is invalid
        """
        name = self._normalize_and_validate_skill_name(name)
        normalized_supporting_files = self._normalize_supporting_files(
            supporting_files,
            allow_deletes=False,
        )
        await self._sync_registry_async(force=True, trash_lock_held=True)
        db = self._get_db()

        existing = db.get_skill_registry(name, include_deleted=True)
        if existing and not existing.get("deleted"):
            raise SkillConflictError(f"Skill '{name}' already exists", skill_name=name)
        if existing and existing.get("deleted") and not replace_deleted:
            raise SkillConflictError(
                f"Skill '{name}' exists in Trash; restore it or permanently delete it first",
                skill_name=name,
            )

        skill_dir = self._get_skill_dir(name)
        staging_dir, parsed = self._prepare_skill_bundle(
            name,
            content,
            normalized_supporting_files,
            operation="create",
        )
        try:
            self._move_skill_dir(staging_dir, skill_dir)
        except FileExistsError:
            self._remove_skill_dir(staging_dir)
            raise SkillConflictError(f"Skill directory '{name}' already exists", skill_name=name) from None
        except OSError as e:
            self._remove_skill_dir(staging_dir)
            raise SkillStorageError(f"Failed to create skill directory: {e}", path=str(skill_dir)) from e

        registry_payload = self._registry_payload(name, skill_dir, parsed)

        try:
            if existing and existing.get("deleted"):
                self._activate_deleted_replacement_locked(
                    name,
                    skill_dir,
                    parsed,
                    existing,
                )
            else:
                db.insert_skill_registry(registry_payload)
        except ConflictError as e:
            self._remove_skill_dir(skill_dir)
            raise SkillConflictError(str(e), skill_name=name) from e
        except (CharactersRAGDBError, InputError) as e:
            self._remove_skill_dir(skill_dir)
            raise SkillsError(f"Failed to record skill '{name}' in registry: {e}") from e

        logger.info(f"Created skill '{name}' for user {self.user_id}")

        return await asyncio.to_thread(
            self._get_skill_sync,
            name,
            enforce_integrity=False,
        )

    async def update_skill(
        self,
        name: str,
        content: Optional[str] = None,
        supporting_files: Optional[dict[str, Optional[str]]] = None,
        expected_version: Optional[int] = None,
    ) -> dict[str, Any]:
        """Update one skill while serializing filesystem and Trash transitions."""
        async with self._trash_operation_lock():
            return await self._finish_trash_mutation(
                self._update_skill_locked(
                    name,
                    content=content,
                    supporting_files=supporting_files,
                    expected_version=expected_version,
                )
            )

    async def _update_skill_locked(
        self,
        name: str,
        content: Optional[str] = None,
        supporting_files: Optional[dict[str, Optional[str]]] = None,
        expected_version: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Update an existing skill.

        Args:
            name: The skill name
            content: New SKILL.md content (optional)
            supporting_files: Files to add/update/remove (set value to None to remove)
            expected_version: Version for optimistic locking

        Returns:
            Updated skill data

        Raises:
            SkillNotFoundError: If skill doesn't exist
            SkillConflictError: If version mismatch
        """
        name = self._normalize_and_validate_skill_name(name)
        await self._sync_registry_async(force=True, trash_lock_held=True)
        db = self._get_db()

        row = db.get_skill_registry(name, include_deleted=False)
        if not row:
            raise SkillNotFoundError(name)

        current_version = int(row.get("version") or 1)
        if expected_version is not None and current_version != expected_version:
            raise SkillConflictError(
                f"Skill '{name}' was modified (expected version {expected_version}, got {current_version})",
                skill_name=name,
                expected_version=expected_version,
                actual_version=current_version,
            )

        skill_dir = self._get_skill_dir(name)
        if not await asyncio.to_thread(skill_dir.exists):
            with contextlib.suppress(Exception):
                db.mark_skill_registry_deleted(name, expected_version=current_version)
            raise SkillNotFoundError(name, detail="Skill directory not found")

        update_data: dict[str, Any] = {}
        touched_files: dict[Path, Optional[str]] = {}

        async def snapshot_file(file_path: Path) -> None:
            """Capture a file's current text content before mutating it."""
            if file_path in touched_files:
                return
            try:
                if await asyncio.to_thread(file_path.exists):
                    touched_files[file_path] = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
                else:
                    touched_files[file_path] = None
            except OSError as e:
                raise SkillStorageError(f"Failed to read existing file before update: {e}", path=str(file_path)) from e

        async def restore_touched_files() -> None:
            """Restore files captured by snapshot_file after a failed update."""
            for file_path, original_content in touched_files.items():
                try:
                    if original_content is None:
                        await asyncio.to_thread(file_path.unlink, missing_ok=True)
                    else:
                        await asyncio.to_thread(file_path.parent.mkdir, parents=True, exist_ok=True)
                        await asyncio.to_thread(file_path.write_text, original_content, encoding="utf-8")
                except OSError as e:
                    logger.error(
                        "Failed to restore file '{}' for skill '{}' and user {} (error {})",
                        file_path.name,
                        name,
                        self.user_id,
                        type(e).__name__,
                    )

        if content is not None:
            try:
                parsed = self._parser.parse_content(content, default_name=name)
            except Exception as e:
                raise SkillValidationError(f"Invalid skill content: {e}") from e
            self._validate_parsed_skill_name(name, parsed)

            skill_file = self._skill_main_file(skill_dir)
            try:
                await snapshot_file(skill_file)
                await asyncio.to_thread(skill_file.write_text, content, encoding="utf-8")
            except SkillStorageError:
                await restore_touched_files()
                raise
            except OSError as e:
                await restore_touched_files()
                raise SkillStorageError(f"Failed to write SKILL.md: {e}", path=str(skill_file)) from e

            update_data.update(
                {
                    "description": parsed.frontmatter.description,
                    "argument_hint": parsed.frontmatter.argument_hint,
                    "disable_model_invocation": parsed.frontmatter.disable_model_invocation,
                    "user_invocable": parsed.frontmatter.user_invocable,
                    "allowed_tools": parsed.frontmatter.allowed_tools,
                    "model": parsed.frontmatter.model,
                    "context": parsed.frontmatter.context,
                    "directory_path": str(skill_dir),
                    "file_hash": parsed.content_hash,
                }
            )

        # Handle supporting files
        if supporting_files:
            normalized_supporting_files = self._normalize_supporting_files(
                supporting_files,
                allow_deletes=True,
            )
            for filename, file_content in normalized_supporting_files.items():
                file_path = self._safe_supporting_path(skill_dir, filename)
                if file_content is None:
                    if await asyncio.to_thread(file_path.exists):
                        try:
                            await snapshot_file(file_path)
                            await asyncio.to_thread(file_path.unlink, missing_ok=True)
                        except SkillStorageError:
                            await restore_touched_files()
                            raise
                        except OSError as e:
                            await restore_touched_files()
                            raise SkillStorageError(
                                f"Failed to delete supporting file '{filename}': {e}",
                                path=str(file_path),
                            ) from e
                else:
                    try:
                        await snapshot_file(file_path)
                        await asyncio.to_thread(file_path.write_text, file_content, encoding="utf-8")
                    except SkillStorageError:
                        await restore_touched_files()
                        raise
                    except OSError as e:
                        await restore_touched_files()
                        raise SkillStorageError(
                            f"Failed to write supporting file '{filename}': {e}",
                            path=str(file_path),
                        ) from e

            if normalized_supporting_files and not update_data:
                parsed = await asyncio.to_thread(self._parse_skill_file, skill_dir)
                if not parsed:
                    await restore_touched_files()
                    raise SkillsError(f"Failed to parse skill '{name}' after supporting file update")
                update_data.update(
                    {
                        "directory_path": str(skill_dir),
                        "file_hash": parsed.content_hash,
                    }
                )

        if update_data:
            try:
                db.update_skill_registry(name, update_data, expected_version=current_version)
            except ConflictError as e:
                await restore_touched_files()
                raise SkillConflictError(str(e), skill_name=name) from e
            except (CharactersRAGDBError, InputError) as e:
                await restore_touched_files()
                raise SkillsError(f"Failed to update skill '{name}' in registry: {e}") from e
            except Exception as e:
                await restore_touched_files()
                raise SkillsError(f"Unexpected error updating skill '{name}' in registry: {e}") from e

        logger.info(f"Updated skill '{name}' for user {self.user_id}")

        return await asyncio.to_thread(
            self._get_skill_sync,
            name,
            enforce_integrity=False,
        )

    async def delete_skill(self, name: str, expected_version: Optional[int] = None) -> None:
        """Move a skill to Trash under the per-user Trash operation lock."""
        async with self._trash_operation_lock():
            await self._finish_trash_mutation(
                self._delete_skill_locked(name, expected_version)
            )

    async def _delete_skill_locked(self, name: str, expected_version: Optional[int] = None) -> None:
        """
        Move a skill bundle to durable Trash and soft-delete its registry row.

        Args:
            name: The skill name
            expected_version: Version for optimistic locking

        Raises:
            SkillNotFoundError: If skill doesn't exist
            SkillConflictError: If version mismatch
        """
        name = name.strip().lower()
        await self._sync_registry_async(force=True, trash_lock_held=True)
        db = self._get_db()

        row = db.get_skill_registry(name, include_deleted=True)
        if not row:
            raise SkillNotFoundError(name)

        current_version = int(row.get("version") or 1)
        if expected_version is not None and current_version != expected_version:
            raise SkillConflictError(
                f"Skill '{name}' was modified (expected version {expected_version}, got {current_version})",
                skill_name=name,
                expected_version=expected_version,
                actual_version=current_version,
            )

        if row.get("deleted"):
            return

        skill_dir = self._get_skill_dir(name)
        archive_dir = self._get_archive_dir(row)
        if not await asyncio.to_thread(skill_dir.exists):
            raise SkillStorageError(
                "Skill bundle is missing and cannot be moved to Trash",
                path=str(skill_dir),
            )
        if await asyncio.to_thread(archive_dir.exists) and not await asyncio.to_thread(
            self._is_skill_bundle_valid,
            name,
            skill_dir,
        ):
            raise SkillStorageError(
                "Skill bundle state is ambiguous; preserved active and archived copies.",
                path=str(skill_dir),
            )

        try:
            await asyncio.to_thread(
                self._discard_stale_archive,
                archive_dir,
                f"stale-{row.get('uuid') or name}",
            )
        except (OSError, SkillStorageError) as e:
            raise SkillStorageError(
                f"Failed to clear a stale Trash archive: {e}",
                path=str(archive_dir),
            ) from e

        try:
            await asyncio.to_thread(self._move_skill_dir, skill_dir, archive_dir)
        except OSError as e:
            raise SkillStorageError(
                f"Failed to move skill to Trash: {e}",
                path=str(skill_dir),
            ) from e

        try:
            db.mark_skill_registry_deleted(
                name,
                expected_version=current_version,
                directory_path=str(archive_dir),
            )
        except ConflictError as e:
            try:
                await asyncio.to_thread(self._move_skill_dir, archive_dir, skill_dir)
            except OSError as rollback_error:
                logger.error(
                    "Failed to roll back conflicted archive for skill '{}' and user {} (error {})",
                    name,
                    self.user_id,
                    type(rollback_error).__name__,
                )
            raise SkillConflictError(str(e), skill_name=name) from e
        except CharactersRAGDBError as e:
            try:
                await asyncio.to_thread(self._move_skill_dir, archive_dir, skill_dir)
            except OSError as rollback_error:
                logger.error(
                    "Failed to roll back archive for skill '{}' and user {} (error {})",
                    name,
                    self.user_id,
                    type(rollback_error).__name__,
                )
            raise SkillsError(f"Failed to delete skill '{name}' in registry: {e}") from e

        self._integrity_decision_cache.pop((self._skill_asset_id(name), "skill_read"), None)
        logger.info(f"Moved skill '{name}' to Trash for user {self.user_id}")

    async def bulk_delete_skills(self, items: list[dict[str, Any]]) -> list[str]:
        """Move selected skills to Trash under one per-user operation lock."""
        async with self._trash_operation_lock():
            return await self._finish_trash_mutation(
                self._bulk_delete_skills_locked(items)
            )

    async def _bulk_delete_skills_locked(self, items: list[dict[str, Any]]) -> list[str]:
        """
        Delete multiple skills after validating all selected versions.

        Args:
            items: Skill name/version mappings. Version may be omitted for legacy rows.

        Returns:
            Deleted skill names in request order.

        Raises:
            SkillValidationError: If the selection is invalid
            SkillNotFoundError: If any selected skill doesn't exist
            SkillConflictError: If any selected version is stale
        """
        if not items:
            raise SkillValidationError("At least one skill is required for bulk delete")

        normalized_items: list[tuple[str, int | None]] = []
        seen_names: set[str] = set()
        for item in items:
            name = self._normalize_and_validate_skill_name(str(item.get("name") or ""))
            if name in seen_names:
                raise SkillValidationError(f"Duplicate skill selected for bulk delete: {name}")
            seen_names.add(name)

            expected_version = item.get("version")
            if expected_version is not None:
                if isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version < 1:
                    raise SkillValidationError(
                        f"Invalid version for skill '{name}'",
                        field="version",
                    )
            normalized_items.append((name, expected_version))

        await self._sync_registry_async(force=True, trash_lock_held=True)
        db = self._get_db()
        move_plans: list[tuple[Path, Path]] = []
        stale_archive_plans: list[tuple[Path, str]] = []
        archive_paths: dict[str, str] = {}
        for name, expected_version in normalized_items:
            row = db.get_skill_registry(name, include_deleted=True)
            if not row:
                raise SkillNotFoundError(name)
            current_version = int(row.get("version") or 1)
            if expected_version is not None and current_version != expected_version:
                raise SkillConflictError(
                    f"Skill '{name}' was modified (expected version {expected_version}, got {current_version})",
                    skill_name=name,
                    expected_version=expected_version,
                    actual_version=current_version,
                )
            if row.get("deleted"):
                continue

            skill_dir = self._get_skill_dir(name)
            if not skill_dir.exists():
                raise SkillStorageError(
                    "Skill bundle is missing and cannot be moved to Trash",
                    path=str(skill_dir),
                )
            archive_dir = self._get_archive_dir(row)
            if archive_dir.exists():
                if not self._is_skill_bundle_valid(name, skill_dir):
                    raise SkillStorageError(
                        "Skill bundle state is ambiguous; preserved active and archived copies.",
                        path=str(skill_dir),
                    )
                stale_archive_plans.append(
                    (archive_dir, f"stale-{row.get('uuid') or name}")
                )
            move_plans.append((skill_dir, archive_dir))
            archive_paths[name] = str(archive_dir)

        for archive_dir, label in stale_archive_plans:
            try:
                await asyncio.to_thread(
                    self._discard_stale_archive,
                    archive_dir,
                    label,
                )
            except (OSError, SkillStorageError) as e:
                raise SkillStorageError(
                    f"Failed to clear a stale Trash archive: {e}",
                    path=str(archive_dir),
                ) from e

        moved: list[tuple[Path, Path]] = []

        async def rollback_moves() -> None:
            for skill_dir, archive_dir in reversed(moved):
                try:
                    await asyncio.to_thread(self._move_skill_dir, archive_dir, skill_dir)
                except OSError as rollback_error:
                    logger.error(
                        "Failed to roll back bulk archive for skill '{}' and user {} (error {})",
                        skill_dir.name,
                        self.user_id,
                        type(rollback_error).__name__,
                    )

        try:
            for skill_dir, archive_dir in move_plans:
                await asyncio.to_thread(self._move_skill_dir, skill_dir, archive_dir)
                moved.append((skill_dir, archive_dir))
        except OSError as e:
            await rollback_moves()
            raise SkillStorageError(
                f"Failed to move selected skills to Trash: {e}",
                path=str(move_plans[len(moved)][0]) if len(moved) < len(move_plans) else None,
            ) from e

        try:
            deleted_rows = await asyncio.to_thread(
                db.bulk_mark_skill_registry_deleted,
                normalized_items,
                archive_paths,
            )
        except InputError as e:
            await rollback_moves()
            message = str(e)
            missing_name = (
                message.removeprefix("Skill not found: ").strip()
                if message.startswith("Skill not found: ")
                else message
            )
            raise SkillNotFoundError(missing_name) from e
        except ConflictError as e:
            await rollback_moves()
            raise SkillConflictError(str(e)) from e
        except CharactersRAGDBError as e:
            await rollback_moves()
            raise SkillsError(f"Failed to bulk delete skills in registry: {e}") from e

        deleted = [str(row["name"]) for row in deleted_rows]
        for name in deleted:
            self._integrity_decision_cache.pop((self._skill_asset_id(name), "skill_read"), None)

        logger.info(f"Moved {len(deleted)} skills to Trash for user {self.user_id}")
        return deleted

    def _build_trash_items(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate archived bundles and build public Trash items off the event loop."""
        items: list[dict[str, Any]] = []
        for row in rows:
            archive_dir = self._get_archive_dir(row)
            has_restore_files = self._is_archive_restorable(archive_dir)
            restorable = has_restore_files and self._is_skill_bundle_valid(
                str(row["name"]),
                archive_dir,
            )
            items.append(
                {
                    "name": row["name"],
                    "description": row.get("description"),
                    "argument_hint": row.get("argument_hint"),
                    "disable_model_invocation": bool(row.get("disable_model_invocation")),
                    "user_invocable": bool(row.get("user_invocable", True)),
                    "allowed_tools": row.get("allowed_tools"),
                    "model": row.get("model"),
                    "context": row.get("context") or "inline",
                    "deleted_at": row.get("last_modified") or datetime.now(timezone.utc),
                    "version": int(row.get("version") or 1),
                    "restorable": restorable,
                    "restore_unavailable_reason": (
                        None
                        if restorable
                        else (
                            "Archived skill files are invalid."
                            if has_restore_files
                            else "Archived skill files are missing."
                        )
                    ),
                }
            )
        return items

    async def list_trash(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """List deleted skills with truthful restore availability."""
        await self._sync_registry_async()
        try:
            rows = await asyncio.to_thread(
                self._get_db().list_deleted_skill_registry,
                limit=limit,
                offset=offset,
            )
        except CharactersRAGDBError as e:
            raise SkillsError(f"Failed to list Skills Trash: {e}") from e
        return await asyncio.to_thread(self._build_trash_items, rows)

    async def get_trash_count(self) -> int:
        """Return the number of records in Skills Trash."""
        await self._sync_registry_async()
        try:
            return await asyncio.to_thread(self._get_db().count_deleted_skill_registry)
        except CharactersRAGDBError as e:
            raise SkillsError(f"Failed to count Skills Trash: {e}") from e

    async def restore_skill(
        self,
        name: str,
        expected_version: Optional[int] = None,
    ) -> dict[str, Any]:
        """Restore a skill under the per-user Trash operation lock."""
        async with self._trash_operation_lock():
            return await self._finish_trash_mutation(
                self._restore_skill_locked(name, expected_version)
            )

    async def _restore_skill_locked(
        self,
        name: str,
        expected_version: Optional[int] = None,
    ) -> dict[str, Any]:
        """Restore a complete skill bundle from durable Trash."""
        name = self._normalize_and_validate_skill_name(name)
        await self._sync_registry_async(force=True, trash_lock_held=True)
        db = self._get_db()
        row = db.get_skill_registry(name, include_deleted=True)
        if not row:
            raise SkillNotFoundError(name)
        if not row.get("deleted"):
            raise SkillConflictError(f"Skill '{name}' is not in Trash", skill_name=name)

        current_version = int(row.get("version") or 1)
        if expected_version is not None and current_version != expected_version:
            raise SkillConflictError(
                f"Skill '{name}' was modified (expected version {expected_version}, got {current_version})",
                skill_name=name,
                expected_version=expected_version,
                actual_version=current_version,
            )

        archive_dir = self._get_archive_dir(row)
        if not self._is_archive_restorable(archive_dir):
            raise SkillConflictError(
                f"Skill '{name}' cannot be restored because archived files are missing",
                skill_name=name,
            )
        skill_dir = self._get_skill_dir(name)
        if skill_dir.exists():
            raise SkillConflictError(
                f"Skill '{name}' cannot be restored because an active directory exists",
                skill_name=name,
            )
        try:
            parsed = await asyncio.to_thread(
                self._parse_unchecked_skill_directory,
                name,
                archive_dir,
            )
            self._validate_parsed_skill_name(name, parsed)
        except Exception:
            raise SkillConflictError(
                f"Skill '{name}' cannot be restored because archived files are invalid",
                skill_name=name,
            ) from None

        try:
            await asyncio.to_thread(self._move_skill_dir, archive_dir, skill_dir)
        except OSError as e:
            raise SkillStorageError(
                f"Failed to restore skill bundle: {e}",
                path=str(archive_dir),
            ) from e

        try:
            db.restore_skill_registry(
                name,
                self._registry_payload(name, skill_dir, parsed),
                expected_version=current_version,
            )
        except ConflictError as e:
            with contextlib.suppress(OSError):
                await asyncio.to_thread(self._move_skill_dir, skill_dir, archive_dir)
            raise SkillConflictError(str(e), skill_name=name) from e
        except CharactersRAGDBError as e:
            with contextlib.suppress(OSError):
                await asyncio.to_thread(self._move_skill_dir, skill_dir, archive_dir)
            raise SkillsError(f"Failed to restore skill '{name}' in registry: {e}") from e

        logger.info(f"Restored skill '{name}' from Trash for user {self.user_id}")
        return await asyncio.to_thread(
            self._get_skill_sync,
            name,
            enforce_integrity=False,
        )

    async def purge_skill(self, name: str, expected_version: Optional[int] = None) -> None:
        """Permanently delete a trashed skill under the per-user operation lock."""
        async with self._trash_operation_lock():
            await self._finish_trash_mutation(
                self._purge_skill_locked(name, expected_version)
            )

    async def _purge_skill_locked(self, name: str, expected_version: Optional[int] = None) -> None:
        """Permanently delete a skill already in Trash."""
        name = self._normalize_and_validate_skill_name(name)
        await self._sync_registry_async(force=True, trash_lock_held=True)
        db = self._get_db()
        row = db.get_skill_registry(name, include_deleted=True)
        if not row:
            raise SkillNotFoundError(name)
        if not row.get("deleted"):
            raise SkillConflictError(f"Skill '{name}' must be moved to Trash first", skill_name=name)

        current_version = int(row.get("version") or 1)
        if expected_version is not None and current_version != expected_version:
            raise SkillConflictError(
                f"Skill '{name}' was modified (expected version {expected_version}, got {current_version})",
                skill_name=name,
                expected_version=expected_version,
                actual_version=current_version,
            )

        archive_dir = self._get_archive_dir(row)
        staging_dir = self.trash_dir / f".purging-{row.get('uuid') or name}"
        staged = False
        if archive_dir.exists():
            try:
                await asyncio.to_thread(self._move_skill_dir, archive_dir, staging_dir)
                staged = True
            except OSError as e:
                raise SkillStorageError(
                    f"Failed to stage skill for permanent deletion: {e}",
                    path=str(archive_dir),
                ) from e

        async def restore_staged_archive() -> None:
            if not staged:
                return
            try:
                await asyncio.to_thread(self._move_skill_dir, staging_dir, archive_dir)
            except OSError as rollback_error:
                logger.error(
                    "Failed to restore purge staging for '{}' and user {}; reconciliation will retry (error {})",
                    name,
                    self.user_id,
                    type(rollback_error).__name__,
                )

        try:
            db.purge_skill_registry(name, expected_version=current_version)
        except InputError as e:
            await restore_staged_archive()
            raise SkillNotFoundError(name) from e
        except ConflictError as e:
            await restore_staged_archive()
            raise SkillConflictError(str(e), skill_name=name) from e
        except CharactersRAGDBError as e:
            await restore_staged_archive()
            raise SkillsError(f"Failed to purge skill '{name}' in registry: {e}") from e

        if staged:
            try:
                cleanup_path = await asyncio.to_thread(
                    self._stage_for_cleanup,
                    staging_dir,
                    f"purged-{row.get('uuid') or name}",
                )
            except (OSError, SkillStorageError) as e:
                logger.warning(
                    "Purge for '{}' is committed for user {}; archive cleanup deferred (error {})",
                    name,
                    self.user_id,
                    type(e).__name__,
                )
            else:
                await asyncio.to_thread(self._remove_cleanup_path_best_effort, cleanup_path)

        logger.info(f"Permanently deleted skill '{name}' for user {self.user_id}")

    async def import_skill(
        self,
        content: str,
        name: Optional[str] = None,
        supporting_files: Optional[dict[str, str]] = None,
        overwrite: bool = False,
        expected_version: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Import a skill from content.

        Args:
            content: SKILL.md content
            name: Override name (otherwise extracted from frontmatter/content)
            supporting_files: Additional files to import
            overwrite: If True, overwrite existing skill
            expected_version: Existing version confirmed by import preview

        Returns:
            Imported skill data
        """
        # Parse content to get name
        try:
            parsed = self._parser.parse_content(content, default_name=name)
        except Exception as e:
            raise SkillValidationError(f"Invalid skill content: {e}") from e

        skill_name, requested_name = self._resolve_import_skill_name(parsed, name)
        if requested_name is not None:
            content = self._parser.rewrite_frontmatter_name(content, requested_name)
        normalized_supporting_files = (
            self._normalize_supporting_files(
                supporting_files,
                allow_deletes=False,
            )
            if supporting_files
            else None
        )

        async with self._trash_operation_lock():
            return await self._finish_trash_mutation(
                self._import_skill_locked(
                    skill_name,
                    content,
                    normalized_supporting_files,
                    overwrite=overwrite,
                    expected_version=expected_version,
                )
            )

    async def _import_skill_locked(
        self,
        skill_name: str,
        content: str,
        supporting_files: Optional[dict[str, str]],
        *,
        overwrite: bool,
        expected_version: Optional[int],
    ) -> dict[str, Any]:
        """Import a validated skill while the per-user operation lock is held."""
        await self._sync_registry_async(force=True, trash_lock_held=True)
        db = self._get_db()
        existing = db.get_skill_registry(skill_name, include_deleted=True)

        if existing:
            current_version = int(existing.get("version") or 1)
            if expected_version is not None and current_version != expected_version:
                raise SkillConflictError(
                    f"Skill '{skill_name}' was modified "
                    f"(expected version {expected_version}, got {current_version})",
                    skill_name=skill_name,
                    expected_version=expected_version,
                    actual_version=current_version,
                )
            if not overwrite:
                location = " in Trash" if existing.get("deleted") else ""
                raise SkillConflictError(
                    f"Skill '{skill_name}' already exists{location}",
                    skill_name=skill_name,
                )
        elif expected_version is not None:
            raise SkillConflictError(
                f"Skill '{skill_name}' no longer exists at the previewed version",
                skill_name=skill_name,
                expected_version=expected_version,
                actual_version=None,
            )

        if existing and not existing.get("deleted"):
            staging_dir, parsed = self._prepare_skill_bundle(
                skill_name,
                content,
                supporting_files or {},
                operation="import-replace",
            )
            try:
                self._replace_active_bundle_locked(
                    skill_name,
                    staging_dir,
                    parsed,
                    existing,
                )
            except ConflictError as e:
                raise SkillConflictError(str(e), skill_name=skill_name) from e
            except (CharactersRAGDBError, InputError) as e:
                raise SkillsError(
                    f"Failed to record imported skill '{skill_name}' in registry: {e}"
                ) from e
            return await asyncio.to_thread(
                self._get_skill_sync,
                skill_name,
                enforce_integrity=False,
            )

        return await self._create_skill_locked(
            skill_name,
            content,
            supporting_files,
            replace_deleted=overwrite,
        )

    def _invalid_import_preview(self, errors: list[str]) -> dict[str, Any]:
        """Build a non-mutating import preview for invalid input."""
        return {
            "valid": False,
            "errors": list(errors),
            "name": None,
            "description": None,
            "argument_hint": None,
            "disable_model_invocation": None,
            "user_invocable": None,
            "allowed_tools": None,
            "model": None,
            "context": None,
            "supporting_file_count": 0,
            "conflict": False,
            "can_overwrite": False,
            "existing_version": None,
        }

    async def preview_import_skill(
        self,
        content: str,
        name: Optional[str] = None,
        supporting_files: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """
        Preview a skill import without creating, deleting, or overwriting files.

        Args:
            content: SKILL.md content
            name: Optional skill name override
            supporting_files: Additional files to validate with the import

        Returns:
            Import review data with parsed metadata, validation errors, and
            conflict state.
        """
        try:
            parsed = self._parser.parse_content(content, default_name=name)
        except SkillParseError as e:
            parse_detail = _public_import_preview_error(e.message) if e.message else ""
            parse_error = f"Invalid skill content: {parse_detail}" if parse_detail else "Invalid skill content"
            return self._invalid_import_preview([parse_error])

        try:
            skill_name, _requested_name = self._resolve_import_skill_name(parsed, name)
            normalized_supporting_files = self._normalize_supporting_files(
                supporting_files,
                allow_deletes=False,
            ) if supporting_files else {}
        except SkillValidationError as e:
            return self._invalid_import_preview([_public_import_preview_error(e.message or "Invalid skill import")])

        await self._sync_registry_async()
        db = self._get_db()
        existing = db.get_skill_registry(skill_name, include_deleted=True)
        has_existing = bool(existing)
        existing_version = int(existing.get("version") or 1) if existing else None
        fm = parsed.frontmatter

        return {
            "valid": True,
            "errors": [],
            "name": skill_name,
            "description": fm.description,
            "argument_hint": fm.argument_hint,
            "disable_model_invocation": fm.disable_model_invocation,
            "user_invocable": fm.user_invocable,
            "allowed_tools": fm.allowed_tools,
            "model": fm.model,
            "context": fm.context,
            "supporting_file_count": len(normalized_supporting_files),
            "conflict": has_existing,
            "can_overwrite": has_existing,
            "existing_version": existing_version,
        }

    def _extract_zip_import_payload(
        self,
        zip_data: bytes,
    ) -> tuple[str, Optional[str], Optional[dict[str, str]]]:
        """Extract SKILL.md content, name, and supporting files from zip bytes."""
        try:
            with zipfile.ZipFile(BytesIO(zip_data), "r") as zf:
                entries = zf.infolist()
                if len(entries) > MAX_ZIP_IMPORT_ENTRIES:
                    raise SkillValidationError(
                        f"Zip file contains too many entries: maximum {MAX_ZIP_IMPORT_ENTRIES} allowed"
                    )
                # Find SKILL.md
                skill_md_info = None
                base_dir = ""

                for info in entries:
                    name = info.filename
                    if name.endswith("SKILL.md"):
                        skill_md_info = info
                        # Get the base directory
                        parts = name.split("/")
                        if len(parts) > 1:
                            base_dir = "/".join(parts[:-1]) + "/"
                        break

                if not skill_md_info:
                    raise SkillValidationError("Zip file does not contain SKILL.md")

                skill_md_path = skill_md_info.filename
                skill_md_posix_path = PurePosixPath(skill_md_path)
                if skill_md_posix_path.is_absolute() or ".." in skill_md_posix_path.parts:
                    raise SkillValidationError(f"Invalid SKILL.md path in zip: '{skill_md_path}'")
                if skill_md_info.file_size > MAX_SKILL_MD_BYTES:
                    raise SkillValidationError("SKILL.md exceeds 500KB limit")

                # Read SKILL.md
                try:
                    content = zf.read(skill_md_path).decode("utf-8")
                except UnicodeDecodeError as e:
                    raise SkillValidationError("SKILL.md in zip must be UTF-8 encoded text") from e

                # Read supporting files
                supporting_files: dict[str, str] = {}
                supporting_count = 0
                supporting_total_bytes = 0
                for info in entries:
                    name = info.filename
                    if name == skill_md_path:
                        continue
                    if name.startswith(base_dir) and not info.is_dir():
                        relative_name = name[len(base_dir) :]
                        if not relative_name:
                            continue

                        relative_path = PurePosixPath(relative_name)
                        if relative_path.is_absolute() or ".." in relative_path.parts:
                            raise SkillValidationError(f"Zip contains path traversal entry: '{name}'")

                        # Ignore nested directories; supporting files are top-level only.
                        if relative_path.name != relative_name:
                            continue

                        safe_filename = self._validate_supporting_filename(relative_name)
                        supporting_count += 1
                        if supporting_count > MAX_SUPPORTING_FILES_COUNT:
                            raise SkillValidationError(
                                f"Too many supporting files: maximum {MAX_SUPPORTING_FILES_COUNT} allowed",
                                field="supporting_files",
                            )
                        if info.file_size > MAX_SUPPORTING_FILE_BYTES:
                            raise SkillValidationError(
                                f"Supporting file '{safe_filename}' exceeds 500KB limit",
                                field="supporting_files",
                            )
                        supporting_total_bytes += info.file_size
                        if supporting_total_bytes > MAX_SUPPORTING_FILES_TOTAL_BYTES:
                            raise SkillValidationError(
                                "Supporting files exceed "
                                f"{MAX_SUPPORTING_FILES_TOTAL_BYTES // (1024 * 1024)}MB limit",
                                field="supporting_files",
                            )
                        try:
                            file_content = zf.read(name).decode("utf-8")
                            supporting_files[safe_filename] = file_content
                        except UnicodeDecodeError:
                            logger.warning(f"Skipping non-text file: {name}")

                # Get skill name from directory
                skill_name = None
                if base_dir:
                    skill_name = base_dir.rstrip("/").split("/")[-1]

                # Validate extracted skill name
                if skill_name:
                    skill_name = self._normalize_and_validate_skill_name(
                        skill_name,
                        source="skill name from zip",
                    )

                return content, skill_name, supporting_files or None

        except zipfile.BadZipFile:
            raise SkillValidationError("Invalid zip file") from None

    async def preview_import_from_zip(
        self,
        zip_data: bytes,
    ) -> dict[str, Any]:
        """
        Preview a skill import from a zip file without mutating stored skills.

        Args:
            zip_data: Zip file bytes

        Returns:
            Import review data
        """
        content, skill_name, supporting_files = self._extract_zip_import_payload(zip_data)
        return await self.preview_import_skill(
            content=content,
            name=skill_name,
            supporting_files=supporting_files,
        )

    async def import_from_zip(
        self,
        zip_data: bytes,
        overwrite: bool = False,
        expected_version: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Import a skill from a zip file.

        Args:
            zip_data: Zip file bytes
            overwrite: If True, overwrite existing skill
            expected_version: Existing version confirmed by import preview

        Returns:
            Imported skill data
        """
        content, skill_name, supporting_files = self._extract_zip_import_payload(zip_data)
        return await self.import_skill(
            content=content,
            name=skill_name,
            supporting_files=supporting_files,
            overwrite=overwrite,
            expected_version=expected_version,
        )

    async def export_skill(self, name: str) -> bytes:
        """
        Export a skill as a zip file.

        Args:
            name: The skill name

        Returns:
            Zip file bytes
        """
        skill_data = await self.get_skill(name)

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write SKILL.md with full content (including frontmatter)
            full_content = skill_data.get("raw_content")
            if not isinstance(full_content, str):
                # Reconstruct from parsed data
                fm = SkillFrontmatter(
                    name=skill_data["name"],
                    description=skill_data.get("description"),
                    argument_hint=skill_data.get("argument_hint"),
                    disable_model_invocation=skill_data.get("disable_model_invocation", False),
                    user_invocable=skill_data.get("user_invocable", True),
                    allowed_tools=skill_data.get("allowed_tools"),
                    model=skill_data.get("model"),
                    context=skill_data.get("context", "inline"),
                )
                full_content = self._parser.serialize_skill(fm, skill_data["content"])

            zf.writestr(f"{name}/SKILL.md", full_content)

            # Write supporting files
            supporting = skill_data.get("supporting_files", {})
            for filename, content in supporting.items():
                zf.writestr(f"{name}/{filename}", content)

        return buffer.getvalue()

    def _build_context_payload(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Build a context payload from skill registry rows."""
        skills = [
            row
            for row in rows
            if row.get("user_invocable")
            and not row.get("disable_model_invocation")
            and self._is_skill_allowed(str(row.get("name") or ""), purpose="skill_context")
        ]

        if not skills:
            return {
                "available_skills": [],
                "context_text": "",
            }

        # Build context text
        lines = ["<available-skills>"]
        for skill in sorted(skills, key=lambda s: s.get("name") or ""):
            hint = f" {skill.get('argument_hint')}" if skill.get("argument_hint") else ""
            desc = skill.get("description") or "No description"
            lines.append(f"- {skill.get('name')}{hint}: {desc}")
        lines.append("</available-skills>")

        return {
            "available_skills": [
                {
                    "name": s.get("name"),
                    "description": s.get("description"),
                    "argument_hint": s.get("argument_hint"),
                    "user_invocable": bool(s.get("user_invocable")),
                    "disable_model_invocation": bool(s.get("disable_model_invocation")),
                    "allowed_tools": s.get("allowed_tools"),
                    "model": s.get("model"),
                    "context": s.get("context") or "inline",
                    "runtime": build_skill_runtime_metadata(
                        context=s.get("context") or "inline",
                        allowed_tools=s.get("allowed_tools"),
                        model=s.get("model"),
                        disable_model_invocation=bool(s.get("disable_model_invocation")),
                    ),
                    "version": int(s.get("version") or 1),
                }
                for s in skills
            ],
            "context_text": "\n".join(lines),
        }

    def _list_context_rows(self) -> list[dict[str, Any]]:
        """List skill registry rows used for context payload generation."""
        db = self._get_db()
        return db.list_skill_registry(
            include_hidden=False,
            include_deleted=False,
            limit=10000,
            offset=0,
        )

    def get_context_payload(self) -> dict[str, Any]:
        """
        Get skill descriptions for context injection.

        Returns a dict with:
        - available_skills: list of skill summaries
        - context_text: formatted text for LLM context
        """
        self._sync_registry()
        rows = self._list_context_rows()
        return self._build_context_payload(rows)

    async def get_context_payload_async(self) -> dict[str, Any]:
        """Async-safe context payload retrieval for async request handlers."""
        await self._sync_registry_async()
        rows = self._list_context_rows()
        return self._build_context_payload(rows)

    async def get_total_count(
        self,
        include_hidden: bool = False,
        q: str | None = None,
        context: str | None = None,
        user_invocable: bool | None = None,
        has_tools: bool | None = None,
        model: str | None = None,
    ) -> int:
        """
        Get total count of skills.

        Args:
            include_hidden: Include skills hidden from user invocation.
            q: Optional search string filtering skills by name or description;
                empty values are ignored.
            context: Optional execution context filter ("inline" or "fork").
            user_invocable: Optional explicit visibility filter.
            has_tools: Optional filter for skills with non-empty allowed_tools.
            model: Optional exact model override filter.
        """
        await self._sync_registry_async()
        db = self._get_db()
        if self.integrity_resolver is not None:
            rows = db.list_skill_registry(
                include_hidden=include_hidden,
                include_deleted=False,
                q=q,
                context=context,
                user_invocable=user_invocable,
                has_tools=has_tools,
                model=model,
                sort="name",
                order="asc",
                limit=None,
                offset=0,
            )
            return sum(
                1 for row in rows if self._is_skill_allowed(str(row.get("name") or ""), purpose="skill_discovery")
            )

        return db.count_skill_registry(
            include_hidden=include_hidden,
            include_deleted=False,
            q=q,
            context=context,
            user_invocable=user_invocable,
            has_tools=has_tools,
            model=model,
        )

    def _get_builtin_skills_dir(self) -> Path:
        """Return the built-in skills source directory."""
        return Path(__file__).parent / "builtin"

    async def seed_builtin_skills(self, overwrite: bool = False) -> list[str]:
        """Seed built-in skills while serializing filesystem and registry changes."""
        async with self._trash_operation_lock():
            return await self._finish_trash_mutation(
                self._seed_builtin_skills_locked(overwrite)
            )

    async def _seed_builtin_skills_locked(self, overwrite: bool) -> list[str]:
        """Copy built-in example skills into the user's skills directory.

        Args:
            overwrite: If True, replace existing skills with same names.

        Returns:
            List of skill names that were seeded.
        """
        builtin_dir = self._get_builtin_skills_dir()
        if not builtin_dir.is_dir():
            logger.warning("Built-in skills directory not found: {}", builtin_dir)
            return []

        await self._sync_registry_async(force=True, trash_lock_held=True)
        db = self._get_db()

        seeded: list[str] = []
        for skill_dir in sorted(builtin_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.is_file():
                continue

            try:
                skill_name = self._normalize_and_validate_skill_name(
                    skill_dir.name,
                    source="built-in skill name",
                )
            except SkillValidationError:
                logger.warning("Skipping built-in skill with invalid name: {}", skill_dir.name)
                continue

            destination_dir = self._get_skill_dir(skill_name)

            try:
                registry_row = db.get_skill_registry(skill_name, include_deleted=True)
            except CharactersRAGDBError as e:
                raise SkillsError(f"Failed to read existing skill state for '{skill_name}': {e}") from e

            row_is_deleted = bool(registry_row and registry_row.get("deleted"))
            skill_exists = bool(registry_row or destination_dir.exists())
            if skill_exists and not overwrite:
                logger.debug("Built-in skill '{}' already exists, skipping", skill_name)
                continue
            if registry_row is None and destination_dir.exists():
                raise SkillStorageError(
                    f"Cannot safely overwrite unregistered skill directory '{skill_name}'",
                    path=str(destination_dir),
                )

            staging_dir = self._new_skill_staging_path(skill_name, "seed")
            try:
                shutil.copytree(skill_dir, staging_dir, dirs_exist_ok=False)
                parsed = self._parse_unchecked_skill_directory(skill_name, staging_dir)
                self._validate_parsed_skill_name(skill_name, parsed)

                if overwrite and registry_row and not row_is_deleted:
                    self._replace_active_bundle_locked(
                        skill_name,
                        staging_dir,
                        parsed,
                        registry_row,
                    )
                elif overwrite and row_is_deleted and registry_row:
                    self._move_skill_dir(staging_dir, destination_dir)
                    self._activate_deleted_replacement_locked(
                        skill_name,
                        destination_dir,
                        parsed,
                        registry_row,
                    )
                else:
                    self._move_skill_dir(staging_dir, destination_dir)
            except ConflictError as e:
                self._remove_skill_dir(staging_dir)
                raise SkillConflictError(str(e), skill_name=skill_name) from e
            except (CharactersRAGDBError, InputError) as e:
                self._remove_skill_dir(staging_dir)
                raise SkillsError(f"Failed to prepare built-in skill '{skill_name}': {e}") from e
            except OSError as e:
                self._remove_skill_dir(staging_dir)
                raise SkillStorageError(
                    f"Failed to copy built-in skill '{skill_name}': {e}",
                    path=str(staging_dir),
                ) from e
            except Exception:
                self._remove_skill_dir(staging_dir)
                raise

            seeded.append(skill_name)
            logger.info("Seeded built-in skill: {}", skill_name)

        if seeded:
            await self._sync_registry_async(force=True, trash_lock_held=True)

        return seeded
