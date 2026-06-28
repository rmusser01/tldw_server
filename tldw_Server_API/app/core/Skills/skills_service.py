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
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Context_Integrity.canonicalization import (
    canonical_filesystem_digest,
)
from tldw_Server_API.app.core.Context_Integrity.resolver import (
    ContextIntegrityBlocked,
    ContextIntegrityResolver,
    get_global_context_integrity_resolver,
)
from tldw_Server_API.app.core.Skills.exceptions import (
    SkillConflictError,
    SkillNotFoundError,
    SkillParseError,
    SkillsError,
    SkillStorageError,
    SkillValidationError,
)
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
SKILL_INTEGRITY_TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".py", ".sh"}
SkillFileFingerprint = tuple[tuple[str, int, int, int, int, int], ...]


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
        self.db = db
        self._parser = SkillParser()
        self._sync_interval = sync_interval
        self._last_sync_time: float = 0.0
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

    def _get_skill_dir(self, name: str) -> Path:
        """Get the directory path for a skill."""
        return self.skills_dir / name

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
        skill_file = skill_dir / "SKILL.md"
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

    def _metadata_from_row(self, row: dict[str, Any]) -> SkillMetadata:
        created_at = row.get("created_at")
        last_modified = row.get("last_modified")
        return SkillMetadata(
            id=row.get("uuid") or row.get("id"),
            name=row.get("name") or "",
            description=row.get("description"),
            argument_hint=row.get("argument_hint"),
            disable_model_invocation=bool(row.get("disable_model_invocation", False)),
            user_invocable=bool(row.get("user_invocable", True)),
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
        """Synchronize skill_registry with filesystem contents.

        Args:
            force: If True, skip debounce and always sync. Write operations
                   should pass force=True.
        """
        now = time.monotonic()
        if not force and (now - self._last_sync_time) < self._sync_interval:
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

                disk_names.add(item.name)
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

    async def _sync_registry_async(self, force: bool = False) -> None:
        """Async wrapper for _sync_registry that offloads filesystem I/O to a thread."""
        now = time.monotonic()
        if not force and (now - self._last_sync_time) < self._sync_interval:
            return
        await asyncio.to_thread(self._sync_registry, force=force)

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

    async def create_skill(
        self,
        name: str,
        content: str,
        supporting_files: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """
        Create a new skill.

        Args:
            name: The skill name (lowercase, hyphens only)
            content: Full SKILL.md content with optional frontmatter
            supporting_files: Additional files to include

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
        await self._sync_registry_async(force=True)
        db = self._get_db()

        existing = db.get_skill_registry(name, include_deleted=True)
        if existing and not existing.get("deleted"):
            raise SkillConflictError(f"Skill '{name}' already exists", skill_name=name)

        # Parse the content to validate
        try:
            parsed = self._parser.parse_content(content, default_name=name)
        except Exception as e:
            raise SkillValidationError(f"Invalid skill content: {e}") from e

        # Create skill directory
        skill_dir = self._get_skill_dir(name)
        try:
            skill_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise SkillConflictError(f"Skill directory '{name}' already exists", skill_name=name) from None
        except OSError as e:
            raise SkillStorageError(f"Failed to create skill directory: {e}", path=str(skill_dir)) from e

        # Write SKILL.md
        skill_file = skill_dir / "SKILL.md"
        try:
            skill_file.write_text(content, encoding="utf-8")
        except OSError as e:
            shutil.rmtree(skill_dir, ignore_errors=True)
            raise SkillStorageError(f"Failed to write SKILL.md: {e}", path=str(skill_file)) from e

        # Write supporting files
        if normalized_supporting_files:
            for filename, file_content in normalized_supporting_files.items():
                file_path = self._safe_supporting_path(skill_dir, filename)
                try:
                    file_path.write_text(file_content or "", encoding="utf-8")
                except OSError as e:
                    shutil.rmtree(skill_dir, ignore_errors=True)
                    raise SkillStorageError(
                        f"Failed to write supporting file {filename}: {e}",
                        path=str(file_path),
                    ) from e

        registry_payload = {
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

        try:
            if existing and existing.get("deleted"):
                # Hard-delete the soft-deleted row first, then fresh insert.
                # update_skill_registry rejects soft-deleted records (WHERE deleted=0),
                # so we purge and re-insert instead.
                db.execute_query(
                    "DELETE FROM skill_registry WHERE name = ? AND deleted = 1",
                    (name,),
                    commit=True,
                )
                db.insert_skill_registry(registry_payload)
            else:
                db.insert_skill_registry(registry_payload)
        except ConflictError as e:
            shutil.rmtree(skill_dir, ignore_errors=True)
            raise SkillConflictError(str(e), skill_name=name) from e
        except (CharactersRAGDBError, InputError) as e:
            shutil.rmtree(skill_dir, ignore_errors=True)
            raise SkillsError(f"Failed to record skill '{name}' in registry: {e}") from e

        logger.info(f"Created skill '{name}' for user {self.user_id}")

        return await self.get_skill(name, enforce_integrity=False)

    async def update_skill(
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
        await self._sync_registry_async(force=True)
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
                    logger.error(f"Failed to restore skill file {file_path}: {e}")

        if content is not None:
            try:
                parsed = self._parser.parse_content(content, default_name=name)
            except Exception as e:
                raise SkillValidationError(f"Invalid skill content: {e}") from e

            skill_file = skill_dir / "SKILL.md"
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

        return await self.get_skill(name, enforce_integrity=False)

    async def delete_skill(self, name: str, expected_version: Optional[int] = None) -> None:
        """
        Delete a skill.

        Args:
            name: The skill name
            expected_version: Version for optimistic locking

        Raises:
            SkillNotFoundError: If skill doesn't exist
            SkillConflictError: If version mismatch
        """
        name = name.strip().lower()
        await self._sync_registry_async(force=True)
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

        if not row.get("deleted"):
            try:
                db.mark_skill_registry_deleted(name, expected_version=current_version)
            except ConflictError as e:
                raise SkillConflictError(str(e), skill_name=name) from e
            except CharactersRAGDBError as e:
                raise SkillsError(f"Failed to delete skill '{name}' in registry: {e}") from e

        # Delete skill directory after the registry accepts the versioned delete.
        skill_dir = self._get_skill_dir(name)
        if await asyncio.to_thread(skill_dir.exists):
            try:
                await asyncio.to_thread(shutil.rmtree, skill_dir)
            except OSError as e:
                if not row.get("deleted"):
                    try:
                        db.restore_skill_registry(
                            name,
                            {"directory_path": str(skill_dir)},
                            expected_version=current_version + 1,
                        )
                    except Exception as restore_error:
                        logger.error(f"Failed to restore skill registry after delete failure: {restore_error}")
                raise SkillStorageError(f"Failed to delete skill directory: {e}", path=str(skill_dir)) from e

        logger.info(f"Deleted skill '{name}' for user {self.user_id}")

    async def import_skill(
        self,
        content: str,
        name: Optional[str] = None,
        supporting_files: Optional[dict[str, str]] = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """
        Import a skill from content.

        Args:
            content: SKILL.md content
            name: Override name (otherwise extracted from frontmatter/content)
            supporting_files: Additional files to import
            overwrite: If True, overwrite existing skill

        Returns:
            Imported skill data
        """
        # Parse content to get name
        try:
            parsed = self._parser.parse_content(content, default_name=name)
        except Exception as e:
            raise SkillValidationError(f"Invalid skill content: {e}") from e

        if parsed.frontmatter.name:
            self._normalize_and_validate_skill_name(parsed.frontmatter.name, source="frontmatter skill name")

        requested_name: Optional[str] = None
        if name is not None:
            requested_name = self._normalize_and_validate_skill_name(name)

        skill_name = requested_name or parsed.frontmatter.name
        if not skill_name:
            raise SkillValidationError("Skill name must be specified in frontmatter or as parameter")

        skill_name = self._normalize_and_validate_skill_name(skill_name)
        normalized_supporting_files = (
            self._normalize_supporting_files(
                supporting_files,
                allow_deletes=False,
            )
            if supporting_files
            else None
        )

        await self._sync_registry_async(force=True)
        db = self._get_db()
        existing = db.get_skill_registry(skill_name, include_deleted=True)

        if existing and not existing.get("deleted"):
            if overwrite:
                await self.delete_skill(skill_name, expected_version=existing.get("version"))
            else:
                raise SkillConflictError(f"Skill '{skill_name}' already exists", skill_name=skill_name)

        return await self.create_skill(skill_name, content, normalized_supporting_files)

    def _invalid_import_preview(self, errors: list[str]) -> dict[str, Any]:
        """Build a non-mutating import preview for invalid input."""
        return {
            "valid": False,
            "errors": errors,
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
            return self._invalid_import_preview([f"Invalid skill content: {e}"])

        try:
            if parsed.frontmatter.name:
                self._normalize_and_validate_skill_name(
                    parsed.frontmatter.name,
                    source="frontmatter skill name",
                )

            requested_name: Optional[str] = None
            if name is not None:
                requested_name = self._normalize_and_validate_skill_name(name)

            skill_name = requested_name or parsed.frontmatter.name
            if not skill_name:
                raise SkillValidationError("Skill name must be specified in frontmatter or as parameter")

            skill_name = self._normalize_and_validate_skill_name(skill_name)
            normalized_supporting_files = self._normalize_supporting_files(
                supporting_files,
                allow_deletes=False,
            ) if supporting_files else {}
        except SkillValidationError as e:
            return self._invalid_import_preview([str(e)])

        await self._sync_registry_async()
        db = self._get_db()
        existing = db.get_skill_registry(skill_name, include_deleted=True)
        active_existing = bool(existing and not existing.get("deleted"))
        existing_version = int(existing.get("version") or 1) if active_existing else None
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
            "conflict": active_existing,
            "can_overwrite": active_existing,
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
    ) -> dict[str, Any]:
        """
        Import a skill from a zip file.

        Args:
            zip_data: Zip file bytes
            overwrite: If True, overwrite existing skill

        Returns:
            Imported skill data
        """
        content, skill_name, supporting_files = self._extract_zip_import_payload(zip_data)
        return await self.import_skill(
            content=content,
            name=skill_name,
            supporting_files=supporting_files,
            overwrite=overwrite,
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
                    "context": s.get("context", "inline"),
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

        await self._sync_registry_async(force=True)
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
            skill_exists = bool((registry_row and not row_is_deleted) or destination_dir.exists())
            try:
                if skill_exists and not overwrite:
                    logger.debug("Built-in skill '{}' already exists, skipping", skill_name)
                    continue

                if overwrite and row_is_deleted:
                    db.execute_query(
                        "DELETE FROM skill_registry WHERE name = ? AND deleted = 1",
                        (skill_name,),
                        commit=True,
                    )

                if overwrite and destination_dir.exists():
                    shutil.rmtree(destination_dir, ignore_errors=True)

                if destination_dir.exists():
                    logger.warning(
                        "Skipping built-in skill '{}' because destination already exists",
                        skill_name,
                    )
                    continue

                shutil.copytree(skill_dir, destination_dir, dirs_exist_ok=False)
            except (CharactersRAGDBError, InputError) as e:
                raise SkillsError(f"Failed to prepare built-in skill '{skill_name}': {e}") from e
            except OSError as e:
                raise SkillStorageError(
                    f"Failed to copy built-in skill '{skill_name}': {e}",
                    path=str(destination_dir),
                ) from e

            seeded.append(skill_name)
            logger.info("Seeded built-in skill: {}", skill_name)

        if seeded:
            await self._sync_registry_async(force=True)

        return seeded
