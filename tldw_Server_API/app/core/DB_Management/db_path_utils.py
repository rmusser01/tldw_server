# db_path_utils.py
"""
Centralized database path management utilities.
Ensures consistent database file locations across the application.
"""

import hashlib
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Optional, Union

from loguru import logger

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.media_db.constants import (
    MEDIA_DB_FILENAME,
)
from tldw_Server_API.app.core.exceptions import InvalidStoragePathError, StorageUnavailableError
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Utils.path_utils import safe_join
from tldw_Server_API.app.core.Utils.Utils import get_project_root

UserId = Union[int, str]
_SAFE_TEST_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_TEST_FALLBACK_RUN_TAG = uuid.uuid4().hex[:8]


def _is_test_context() -> bool:
    return bool(os.getenv("PYTEST_CURRENT_TEST")) or is_test_mode()


def _get_test_fallback_run_tag() -> str:
    explicit_tag = str(os.getenv("TLDW_TEST_RUN_ID") or "").strip()
    if explicit_tag:
        return explicit_tag

    worker_tag = str(os.getenv("PYTEST_XDIST_WORKER") or "").strip() or "default"
    return f"{worker_tag}-{_TEST_FALLBACK_RUN_TAG}"


def _normalize_user_id(user_id: UserId) -> str:
    raw = str(user_id).strip()
    if not raw:
        raise ValueError("user_id must not be empty for filesystem path")
    if raw.isdigit():
        if int(raw) < 1:
            raise ValueError(f"user_id must be a positive integer for filesystem path: {user_id!r}")
        return raw
    if _is_test_context():
        if (
            _SAFE_TEST_USER_ID_RE.fullmatch(raw)
            and raw[0].isalnum()
            and raw[-1].isalnum()
        ):
            return raw
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"user_{digest}"
    raise ValueError(f"Invalid user_id for filesystem path: {user_id!r}")


def _get_auth_mode() -> str:
    raw = os.getenv("AUTH_MODE")
    if raw:
        return raw.strip().lower()
    try:
        from_settings = settings.get("AUTH_MODE")
    except Exception:
        from_settings = None
    if not from_settings:
        return "single_user"
    return str(from_settings).strip().lower() or "single_user"


def _is_single_user_mode() -> bool:
    return _get_auth_mode() == "single_user"


def _resolve_user_id_for_storage(user_id: Optional[UserId]) -> str:
    if user_id is None:
        if _is_single_user_mode():
            return _normalize_user_id(DatabasePaths.get_single_user_id())
        raise ValueError("user_id is required in multi-user mode")
    raw = str(user_id).strip()
    if not raw:
        if _is_single_user_mode():
            return _normalize_user_id(DatabasePaths.get_single_user_id())
        raise ValueError("user_id is required in multi-user mode")
    return _normalize_user_id(raw)


def _normalize_user_db_base_dir(raw_path: Path) -> Path:
    try:
        candidate = raw_path.expanduser()
    except Exception:
        candidate = raw_path

    if not candidate.is_absolute():
        project_root = Path(get_project_root()).resolve()
        candidate = (project_root / candidate).resolve()
        try:
            candidate.relative_to(project_root)
        except ValueError as exc:
            raise InvalidStoragePathError("invalid_path") from exc
    else:
        candidate = candidate.resolve()
    return candidate


def _ensure_dir(path: Path, *, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create {label} directory {path}: {e}")
        raise StorageUnavailableError(f"Failed to create {label} directory") from e


def _resolve_candidate_for_containment(candidate: Path) -> Path:
    parent = candidate.parent.resolve(strict=False)
    return parent / candidate.name


def resolve_trusted_database_path(
    db_path: str | Path,
    *,
    label: str = "database",
    extra_roots: Optional[list[Path]] = None,
) -> Path:
    """Resolve a database path and require containment within trusted roots."""
    raw_path = str(db_path or "").strip()
    if not raw_path:
        raise InvalidStoragePathError("invalid_path")

    trusted_roots: list[Path] = []
    project_root = Path(get_project_root()).resolve()
    trusted_roots.append(project_root)

    try:
        user_db_base_dir = DatabasePaths.get_user_db_base_dir(allow_legacy_alias=True).resolve()
    except Exception as exc:
        logger.debug("Skipping user database trusted root discovery: {}", exc)
    else:
        trusted_roots.append(user_db_base_dir)

    if extra_roots:
        for root in extra_roots:
            try:
                resolved_root = Path(root).expanduser().resolve(strict=False)
            except Exception as exc:
                logger.debug("Skipping invalid trusted root {}: {}", root, exc)
            else:
                trusted_roots.append(resolved_root)

    if _is_test_context():
        trusted_roots.append(Path(tempfile.gettempdir()).resolve())
        trusted_roots.append(Path.cwd().resolve())

    unique_roots: list[Path] = []
    for root in trusted_roots:
        if root not in unique_roots:
            unique_roots.append(root)

    try:
        expanded_path = os.path.expanduser(raw_path)
    except Exception as exc:
        raise InvalidStoragePathError("invalid_path") from exc

    if os.path.isabs(expanded_path):
        normalized_absolute = os.path.normpath(expanded_path)
        for root in unique_roots:
            root_resolved = root.resolve(strict=False)
            try:
                relative_candidate = os.path.relpath(normalized_absolute, str(root_resolved))
            except ValueError:
                continue

            safe_candidate = safe_join(str(root_resolved), relative_candidate)
            if safe_candidate is not None:
                return Path(safe_candidate)

        logger.warning("Rejected {} path outside trusted roots: {}", label, normalized_absolute)
        raise InvalidStoragePathError("invalid_path")

    safe_candidate = safe_join(str(project_root), expanded_path)
    if safe_candidate is not None:
        return Path(safe_candidate)

    logger.warning("Rejected {} path outside trusted roots: {}", label, expanded_path)
    raise InvalidStoragePathError("invalid_path")


def _build_user_dir(base_path: Path, user_id: Optional[UserId]) -> Path:
    safe_user_id = _resolve_user_id_for_storage(user_id)
    user_dir = (base_path / safe_user_id).resolve()
    try:
        user_dir.relative_to(base_path)
    except ValueError as exc:
        raise ValueError(f"Computed user directory escapes base path: {user_dir!r}") from exc
    _ensure_dir(user_dir, label="user")
    return user_dir


def normalize_output_storage_filename(
    storage_path: str | Path,
    *,
    allow_absolute: bool,
    reject_relative_with_separators: bool,
    expand_user: bool = False,
    base_resolved: Optional[Path] = None,
    check_relative_containment: bool = False,
    require_parent_base: bool = False,
    log_message: Optional[Callable[[str], None]] = None,
    log_prefix: Optional[str] = None,
) -> str:
    """Normalize and validate an output storage path, returning a safe filename."""
    def _log(message: str) -> None:
        if log_message is None:
            return
        if log_prefix:
            log_message(f"{log_prefix}: {message}")
        else:
            log_message(message)

    def _fail(message: str, exc: Optional[Exception] = None) -> None:
        _log(message)
        if exc is None:
            raise InvalidStoragePathError("invalid_path")
        raise InvalidStoragePathError("invalid_path") from exc

    storage_path_str = str(storage_path)

    if not check_relative_containment and isinstance(storage_path, str) and (
        _SAFE_OUTPUT_NAME_RE.match(storage_path)
        and os.sep not in storage_path
        and (os.altsep is None or os.altsep not in storage_path)
    ):
        return storage_path

    candidate = Path(storage_path_str)
    if expand_user:
        candidate = candidate.expanduser()

    if candidate.is_absolute():
        if not allow_absolute:
            _fail(f"absolute paths are not allowed for outputs: {candidate}")
    elif reject_relative_with_separators and (
        os.sep in storage_path_str or (os.altsep and os.altsep in storage_path_str)
    ):
        _fail("nested output paths are not allowed")

    candidate_name = candidate.name
    if not candidate_name:
        _fail(f"empty output path component from {storage_path!r}")
    if os.sep in candidate_name or (os.altsep and os.altsep in candidate_name):
        _fail(f"path separator detected in output filename: {candidate_name!r}")
    if not _SAFE_OUTPUT_NAME_RE.match(candidate_name):
        _fail(f"invalid characters in output filename: {candidate_name!r}")

    if candidate.is_absolute():
        if base_resolved is None:
            _fail("outputs base directory unavailable for absolute path validation")
        try:
            resolved = candidate.resolve(strict=False)
        except Exception as exc:
            _fail(f"invalid output path {storage_path}: {exc}", exc)
        if not resolved.is_relative_to(base_resolved):
            _fail(f"output path outside base dir: {resolved}")
        if require_parent_base and resolved.parent != base_resolved:
            _fail(f"output path outside base dir: {resolved}")
    elif check_relative_containment:
        if base_resolved is None:
            _fail("outputs base directory unavailable for path validation")
        try:
            resolved = (base_resolved / candidate_name).resolve(strict=False)
        except Exception as exc:
            _fail(f"invalid output path {storage_path}: {exc}", exc)
        if not resolved.is_relative_to(base_resolved):
            _fail(f"output path outside base dir: {resolved}")

    return candidate_name



class DatabasePaths:
    """Centralized database path management."""

    # Database file names
    MEDIA_DB_NAME = MEDIA_DB_FILENAME
    CHACHA_DB_NAME = "ChaChaNotes.db"
    PROMPTS_DB_NAME = "user_prompts_v2.sqlite"
    AUDIT_DB_NAME = "unified_audit.db"
    EVALUATIONS_DB_NAME = "evaluations.db"
    PERSONALIZATION_DB_NAME = "Personalization.db"
    GUARDIAN_DB_NAME = "Guardian.db"
    WORKFLOWS_DB_NAME = "workflows.db"
    WORKFLOWS_SCHEDULER_DB_NAME = "workflows_scheduler.db"
    RESEARCH_SESSIONS_DB_NAME = "ResearchSessions.db"
    KANBAN_DB_NAME = "Kanban.db"
    SLIDES_DB_NAME = "Slides.db"
    VOICE_REGISTRY_DB_NAME = "voice_registry.db"
    CIRCUIT_BREAKER_REGISTRY_DB_NAME = "circuit_breaker_registry.db"

    # Subdirectories
    PROMPTS_SUBDIR = "prompts_user_dbs"
    AUDIT_SUBDIR = "audit"
    EVALUATIONS_SUBDIR = "evaluations"
    WORKFLOWS_SUBDIR = "workflows"
    PROMPT_STUDIO_SUBDIR = "prompt_studio_dbs"
    OUTPUTS_SUBDIR = "outputs"
    OUTPUTS_TEMP_SUBDIR = "outputs_tmp"
    CHROMA_SUBDIR = "chroma_storage"
    VECTOR_STORE_SUBDIR = "vector_store"
    VOICES_SUBDIR = "voices"
    REWRITE_CACHE_SUBDIR = "Rewrite_Cache"
    CHATBOOKS_SUBDIR = "chatbooks"
    CHATBOOKS_EXPORTS_SUBDIR = "exports"
    CHATBOOKS_IMPORTS_SUBDIR = "imports"
    CHATBOOKS_TEMP_SUBDIR = "temp"
    READING_IMPORTS_SUBDIR = "reading_imports"

    @staticmethod
    def get_user_db_base_dir(*, allow_legacy_alias: bool = False) -> Path:
        env_user_db_base = os.getenv("USER_DB_BASE_DIR")
        settings_user_db_base = settings.get("USER_DB_BASE_DIR")
        project_root = Path(get_project_root())
        default_base = (project_root / "Databases" / "user_databases").resolve()
        user_db_base = settings_user_db_base or env_user_db_base
        if _is_test_context() and env_user_db_base:
            try:
                settings_candidate = Path(settings_user_db_base) if settings_user_db_base else None
                if settings_candidate is not None:
                    settings_candidate = settings_candidate.expanduser()
                    if not settings_candidate.is_absolute():
                        settings_candidate = (project_root / settings_candidate).resolve()
                    else:
                        settings_candidate = settings_candidate.resolve()
            except Exception:
                settings_candidate = None
            if settings_candidate is None or settings_candidate == default_base:
                user_db_base = env_user_db_base
        if _is_test_context() and not env_user_db_base:
            try:
                candidate = Path(settings_user_db_base) if settings_user_db_base else None
                if candidate is not None:
                    candidate = candidate.expanduser()
                    if not candidate.is_absolute():
                        candidate = (project_root / candidate).resolve()
                    else:
                        candidate = candidate.resolve()
            except Exception:
                candidate = None
            if candidate is None or candidate == default_base:
                user_db_base = None
        if not user_db_base and allow_legacy_alias:
            legacy_base = os.getenv("USER_DB_BASE") or settings.get("USER_DB_BASE")
            if legacy_base:
                logger.warning(
                    "USER_DB_BASE is deprecated; use USER_DB_BASE_DIR instead. "
                    "Rewrite cache will stop honoring USER_DB_BASE in a future release."
                )
                user_db_base = legacy_base
        if not user_db_base:
            if _is_test_context():
                import tempfile

                run_tag = _get_test_fallback_run_tag()
                safe_run_tag = "".join(
                    ch if ch.isalnum() or ch in "-_." else "_"
                    for ch in str(run_tag)
                )
                base_path = (
                    Path(tempfile.gettempdir()) / "tldw_user_databases_test" / safe_run_tag
                ).resolve()
                logger.warning(
                    f"USER_DB_BASE_DIR not configured in tests, using isolated fallback: {base_path}"
                )
            else:
                base_path = (project_root / "Databases" / "user_databases").resolve()
                logger.warning(f"USER_DB_BASE_DIR not configured, using fallback: {base_path}")
        else:
            base_path = _normalize_user_db_base_dir(Path(user_db_base))
        _ensure_dir(base_path, label="user database base")
        return base_path

    @staticmethod
    def get_user_base_directory(
        user_id: Optional[UserId],
        *,
        base_dir_override: Optional[Union[str, Path]] = None,
        allow_legacy_alias: bool = False,
    ) -> Path:
        """
        Get the base directory for a specific user's databases.

        Args:
            user_id: The user's ID
            base_dir_override: Optional base directory override
            allow_legacy_alias: Whether to allow deprecated USER_DB_BASE

        Returns:
            Path to the user's database directory
        """
        if base_dir_override is not None:
            base_path = _normalize_user_db_base_dir(Path(base_dir_override))
            _ensure_dir(base_path, label="user database base")
        else:
            base_path = DatabasePaths.get_user_db_base_dir(allow_legacy_alias=allow_legacy_alias)
        user_dir = _build_user_dir(base_path, user_id)
        logger.debug(f"Ensured user directory exists: {user_dir}")
        return user_dir

    @staticmethod
    def get_media_db_path(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's media database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        return user_dir / DatabasePaths.MEDIA_DB_NAME

    @staticmethod
    def get_chacha_db_path(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's ChaChaNotes database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        return user_dir / DatabasePaths.CHACHA_DB_NAME

    @staticmethod
    def get_prompts_db_path(user_id: Optional[UserId], *, salt: Optional[str] = None) -> Path:
        """Get the path to the user's prompts database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        prompts_dir = user_dir / DatabasePaths.PROMPTS_SUBDIR

        # Ensure prompts subdirectory exists
        _ensure_dir(prompts_dir, label="prompts")

        if salt:
            safe_salt = normalize_output_storage_filename(
                salt,
                allow_absolute=False,
                reject_relative_with_separators=True,
                expand_user=False,
            )
            return prompts_dir / f"user_prompts_v2_{safe_salt}.sqlite"
        return prompts_dir / DatabasePaths.PROMPTS_DB_NAME

    @staticmethod
    def get_audit_db_path(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's audit database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        audit_dir = user_dir / DatabasePaths.AUDIT_SUBDIR

        # Ensure audit subdirectory exists
        _ensure_dir(audit_dir, label="audit")

        return audit_dir / DatabasePaths.AUDIT_DB_NAME

    @staticmethod
    def get_shared_audit_db_path() -> Path:
        """Get the path to the shared audit database."""
        raw = os.getenv("AUDIT_SHARED_DB_PATH") or settings.get("AUDIT_SHARED_DB_PATH")
        if raw:
            try:
                candidate = Path(str(raw)).expanduser()
            except Exception:
                candidate = Path(str(raw))
            if not candidate.is_absolute():
                project_root = Path(get_project_root())
                candidate = (project_root / candidate).resolve()
            else:
                candidate = candidate.resolve()
        else:
            project_root = Path(get_project_root())
            candidate = (project_root / "Databases" / "audit_shared.db").resolve()
        _ensure_dir(candidate.parent, label="shared audit")
        return candidate

    @staticmethod
    def get_shared_circuit_breaker_db_path() -> Path:
        """Get the path to the shared circuit breaker registry database."""
        raw = (
            os.getenv("CIRCUIT_BREAKER_REGISTRY_DB_PATH")
            or settings.get("CIRCUIT_BREAKER_REGISTRY_DB_PATH")
        )
        if raw:
            try:
                candidate = Path(str(raw)).expanduser()
            except Exception:
                candidate = Path(str(raw))
            if not candidate.is_absolute():
                project_root = Path(get_project_root())
                candidate = (project_root / candidate).resolve()
            else:
                candidate = candidate.resolve()
        else:
            project_root = Path(get_project_root())
            candidate = (
                project_root
                / "Databases"
                / DatabasePaths.CIRCUIT_BREAKER_REGISTRY_DB_NAME
            ).resolve()
        _ensure_dir(candidate.parent, label="shared circuit breaker")
        return candidate

    @staticmethod
    def get_evaluations_db_path(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's evaluations database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        eval_dir = user_dir / DatabasePaths.EVALUATIONS_SUBDIR

        # Ensure evaluations subdirectory exists
        _ensure_dir(eval_dir, label="evaluations")

        return eval_dir / DatabasePaths.EVALUATIONS_DB_NAME

    @staticmethod
    def get_personalization_db_path(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's Personalization database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        # Keep at root of user dir alongside ChaChaNotes for discoverability
        return user_dir / DatabasePaths.PERSONALIZATION_DB_NAME

    @staticmethod
    def get_guardian_db_path(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's Guardian/self-monitoring database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        return user_dir / DatabasePaths.GUARDIAN_DB_NAME

    @staticmethod
    def get_workflows_db_path(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's workflows database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        workflows_dir = user_dir / DatabasePaths.WORKFLOWS_SUBDIR
        _ensure_dir(workflows_dir, label="workflows")
        return workflows_dir / DatabasePaths.WORKFLOWS_DB_NAME

    @staticmethod
    def get_chat_workflows_db_path(user_id: Optional[UserId]) -> Path:
        """Backward-compatible alias for the per-user workflows database path."""
        return DatabasePaths.get_workflows_db_path(user_id)

    @staticmethod
    def get_workflows_scheduler_db_path(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's workflows scheduler database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        workflows_dir = user_dir / DatabasePaths.WORKFLOWS_SUBDIR
        _ensure_dir(workflows_dir, label="workflows scheduler")
        return workflows_dir / DatabasePaths.WORKFLOWS_SCHEDULER_DB_NAME

    @staticmethod
    def get_research_sessions_db_path(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's research sessions database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        return user_dir / DatabasePaths.RESEARCH_SESSIONS_DB_NAME

    @staticmethod
    def get_kanban_db_path(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's Kanban database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        # Keep at root of user dir alongside ChaChaNotes for discoverability
        return user_dir / DatabasePaths.KANBAN_DB_NAME

    @staticmethod
    def get_slides_db_path(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's Slides database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        return user_dir / DatabasePaths.SLIDES_DB_NAME

    @staticmethod
    def get_prompt_studio_db_path(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's Prompt Studio database."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        studio_dir = user_dir / DatabasePaths.PROMPT_STUDIO_SUBDIR
        _ensure_dir(studio_dir, label="prompt studio")
        return studio_dir / "prompt_studio.db"

    @staticmethod
    def get_user_outputs_dir(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's outputs directory."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        outputs_dir = user_dir / DatabasePaths.OUTPUTS_SUBDIR
        _ensure_dir(outputs_dir, label="outputs")
        return outputs_dir

    @staticmethod
    def get_user_temp_outputs_dir(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's transient outputs directory."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        outputs_dir = user_dir / DatabasePaths.OUTPUTS_TEMP_SUBDIR
        _ensure_dir(outputs_dir, label="temp outputs")
        return outputs_dir

    @staticmethod
    def get_user_chroma_dir(
        user_id: Optional[UserId],
        *,
        base_dir_override: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Get the path to the user's ChromaDB storage directory."""
        user_dir = DatabasePaths.get_user_base_directory(user_id, base_dir_override=base_dir_override)
        chroma_dir = user_dir / DatabasePaths.CHROMA_SUBDIR
        _ensure_dir(chroma_dir, label="chroma storage")
        return chroma_dir

    @staticmethod
    def get_user_vector_store_dir(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's vector store directory."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        vector_dir = user_dir / DatabasePaths.VECTOR_STORE_SUBDIR
        _ensure_dir(vector_dir, label="vector store")
        return vector_dir

    @staticmethod
    def get_user_voices_dir(
        user_id: Optional[UserId],
        *,
        base_dir_override: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Get the path to the user's voices directory (including subdirs)."""
        user_dir = DatabasePaths.get_user_base_directory(
            user_id,
            base_dir_override=base_dir_override,
        )
        voices_dir = user_dir / DatabasePaths.VOICES_SUBDIR
        _ensure_dir(voices_dir, label="voices")
        _ensure_dir(voices_dir / "uploads", label="voice uploads")
        _ensure_dir(voices_dir / "processed", label="voice processed")
        _ensure_dir(voices_dir / "temp", label="voice temp")
        _ensure_dir(voices_dir / "metadata", label="voice metadata")
        return voices_dir

    @staticmethod
    def get_user_voice_registry_db_path(
        user_id: Optional[UserId],
        *,
        base_dir_override: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Get the path to the user's persistent voice registry database."""
        voices_dir = DatabasePaths.get_user_voices_dir(
            user_id,
            base_dir_override=base_dir_override,
        )
        return voices_dir / DatabasePaths.VOICE_REGISTRY_DB_NAME

    @staticmethod
    def get_user_chatbooks_dir(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's chatbooks directory."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        chatbooks_dir = user_dir / DatabasePaths.CHATBOOKS_SUBDIR
        _ensure_dir(chatbooks_dir, label="chatbooks")
        return chatbooks_dir

    @staticmethod
    def get_user_chatbooks_exports_dir(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's chatbooks exports directory."""
        chatbooks_dir = DatabasePaths.get_user_chatbooks_dir(user_id)
        exports_dir = chatbooks_dir / DatabasePaths.CHATBOOKS_EXPORTS_SUBDIR
        _ensure_dir(exports_dir, label="chatbooks exports")
        return exports_dir

    @staticmethod
    def get_user_chatbooks_imports_dir(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's chatbooks imports directory."""
        chatbooks_dir = DatabasePaths.get_user_chatbooks_dir(user_id)
        imports_dir = chatbooks_dir / DatabasePaths.CHATBOOKS_IMPORTS_SUBDIR
        _ensure_dir(imports_dir, label="chatbooks imports")
        return imports_dir

    @staticmethod
    def get_user_chatbooks_temp_dir(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's chatbooks temp directory."""
        chatbooks_dir = DatabasePaths.get_user_chatbooks_dir(user_id)
        temp_dir = chatbooks_dir / DatabasePaths.CHATBOOKS_TEMP_SUBDIR
        _ensure_dir(temp_dir, label="chatbooks temp")
        return temp_dir

    @staticmethod
    def get_user_reading_imports_dir(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's reading import staging directory."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        imports_dir = user_dir / DatabasePaths.READING_IMPORTS_SUBDIR
        _ensure_dir(imports_dir, label="reading imports")
        return imports_dir

    @staticmethod
    def get_user_rewrite_cache_path(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's rewrite cache file."""
        base_path = DatabasePaths.get_user_db_base_dir(allow_legacy_alias=True)
        user_dir = _build_user_dir(base_path, user_id)
        cache_dir = user_dir / DatabasePaths.REWRITE_CACHE_SUBDIR
        _ensure_dir(cache_dir, label="rewrite cache")
        return cache_dir / "rewrite_cache.jsonl"

    @staticmethod
    def get_user_rag_personalization_path(user_id: Optional[UserId]) -> Path:
        """Get the path to the user's RAG personalization file."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        return user_dir / "rag_personalization.json"

    @staticmethod
    def get_all_user_db_paths(user_id: Optional[UserId]) -> dict[str, Path]:
        """
        Get all database paths for a user.

        Returns:
            Dictionary mapping database types to their paths
        """
        return {
            "media": DatabasePaths.get_media_db_path(user_id),
            "chacha": DatabasePaths.get_chacha_db_path(user_id),
            "prompts": DatabasePaths.get_prompts_db_path(user_id),
            "audit": DatabasePaths.get_audit_db_path(user_id),
            "evaluations": DatabasePaths.get_evaluations_db_path(user_id),
            "personalization": DatabasePaths.get_personalization_db_path(user_id),
            "workflows": DatabasePaths.get_workflows_db_path(user_id),
            "workflows_scheduler": DatabasePaths.get_workflows_scheduler_db_path(user_id),
            "research_sessions": DatabasePaths.get_research_sessions_db_path(user_id),
            "kanban": DatabasePaths.get_kanban_db_path(user_id),
            "slides": DatabasePaths.get_slides_db_path(user_id),
        }

    @staticmethod
    def validate_database_structure(user_id: Optional[UserId]) -> bool:
        """
        Validate that all required directories exist for a user.

        Args:
            user_id: The user's ID

        Returns:
            True if all directories exist or were created successfully
        """
        try:
            DatabasePaths.get_all_user_db_paths(user_id)
            logger.info(f"Database structure validated for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to validate database structure for user {user_id}: {e}")
            return False

    @staticmethod
    def get_single_user_id() -> int:
        """
        Get the user ID for single-user mode.

        Returns:
            The configured single-user ID (typically 1)
        """
        return int(settings.get("SINGLE_USER_FIXED_ID", "1"))


# Convenience functions for backward compatibility
def get_user_media_db_path(user_id: Optional[UserId]) -> str:
    """Get the media database path for a user (returns string for compatibility)."""
    return str(DatabasePaths.get_media_db_path(user_id))


def get_user_chacha_db_path(user_id: Optional[UserId]) -> str:
    """Get the ChaChaNotes database path for a user (returns string for compatibility)."""
    return str(DatabasePaths.get_chacha_db_path(user_id))


def get_user_prompts_db_path(user_id: Optional[UserId]) -> str:
    """Get the prompts database path for a user (returns string for compatibility)."""
    return str(DatabasePaths.get_prompts_db_path(user_id))


def ensure_user_database_structure(user_id: Optional[UserId]) -> bool:
    """
    Ensure all database directories exist for a user.

    Args:
        user_id: The user's ID

    Returns:
        True if successful
    """
    return DatabasePaths.validate_database_structure(user_id)
