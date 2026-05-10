from __future__ import annotations

# chatbook_service.py
# Description: Service for creating and importing chatbooks with multi-user support
# Adapted from single-user to multi-user architecture
#
"""
Chatbook Service for Multi-User Environment
--------------------------------------------

Handles the creation, import, and export of chatbooks with user isolation.

Key Adaptations from Single-User:
- User-specific exports with access control
- Job-based operations for async processing
- Temporary storage with automatic cleanup
- Per-user database isolation
- No global state or singletons
"""

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

import aiofiles
import aiofiles.os
from loguru import logger

from tldw_Server_API.app.core.config import load_comprehensive_config, settings as core_settings
from tldw_Server_API.app.core.testing import is_truthy

from ..DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from ..DB_Management.db_path_utils import DatabasePaths
from ..Templating.template_renderer import (
    TemplateContext,
    TemplateEnv,
    options_from_env,
)
from ..Templating.template_renderer import render as render_template

# Legacy job queue shim removed; using in-process task registry
from .chatbook_models import (
    ChatbookContent,
    ChatbookManifest,
    ChatbookVersion,
    ConflictResolution,
    ContentItem,
    ContentType,
    ExportJob,
    ExportStatus,
    ImportJob,
    ImportStatus,
    ImportStatusData,
)

# Unified audit logging is handled at the API layer. The service no longer
# imports or depends on legacy audit loggers.
# Import custom exceptions
from .exceptions import (
    ArchiveError,
    DatabaseError,
    ExportError,
    FileOperationError,
    JobError,
    SecurityError,
    ValidationError,
)
from .import_adapters.openwebui import (
    OpenWebUIConversationPlan,
    OpenWebUIMessagePlan,
    load_openwebui_export,
    preview_openwebui_export,
)
from .import_adapters.openwebui_db import (
    extract_openwebui_db_user,
    preview_openwebui_db as build_openwebui_db_preview,
)
from .openwebui_folders import (
    build_openwebui_namespace_segments,
    mirror_openwebui_folder_for_conversation,
)

_CHATBOOK_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ArchiveError,
    DatabaseError,
    ExportError,
    FileOperationError,
    JobError,
    SecurityError,
    ValidationError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    json.JSONDecodeError,
    zipfile.BadZipFile,
    asyncio.CancelledError,
)
_OPENWEBUI_FOLDER_MIRROR_EXCEPTIONS: tuple[type[BaseException], ...] = (
    *_CHATBOOK_NONCRITICAL_EXCEPTIONS,
    CharactersRAGDBError,
)

_CHATBOOK_TEMPLATE_MODES = {"pass_through", "render_on_export", "render_on_import"}

try:  # Prompts database is optional in some deployments
    from ..DB_Management.Prompts_DB import PromptsDatabase  # type: ignore
except _CHATBOOK_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - defensive guard for stripped builds
    PromptsDatabase = None  # type: ignore

try:
    from ..DB_Management.media_db.api import (  # type: ignore
        create_media_database,
        get_media_by_id,
        get_media_prompts,
        get_media_transcripts,
        get_media_by_uuid,
    )
except _CHATBOOK_NONCRITICAL_EXCEPTIONS:  # pragma: no cover
    create_media_database = None  # type: ignore
    get_media_by_id = None  # type: ignore
    get_media_by_uuid = None  # type: ignore
    get_media_transcripts = None  # type: ignore
    get_media_prompts = None  # type: ignore

try:
    from ..DB_Management.Evaluations_DB import EvaluationsDatabase  # type: ignore
except _CHATBOOK_NONCRITICAL_EXCEPTIONS:  # pragma: no cover
    EvaluationsDatabase = None  # type: ignore

try:
    from ..Embeddings.ChromaDB_Library import ChromaDBManager  # type: ignore
except _CHATBOOK_NONCRITICAL_EXCEPTIONS:  # pragma: no cover
    ChromaDBManager = None  # type: ignore


class ChatbookService:
    """Service for creating and importing chatbooks with user isolation."""

    @staticmethod
    def _is_unsafe_archive_path(member_path: str) -> bool:
        """
        Check if an archive member path is potentially unsafe (path traversal).

        This performs path-component aware checking to detect:
        - Absolute paths
        - Parent directory references (..)
        - Paths that could escape the extraction directory

        Args:
            member_path: The path of a member within the archive

        Returns:
            True if the path is unsafe, False otherwise
        """
        # Normalize the path first
        normalized = os.path.normpath(member_path)

        # Check for absolute paths (Unix or Windows style)
        if os.path.isabs(normalized) or normalized.startswith("/") or normalized.startswith("\\"):
            return True

        # Check for Windows drive letters (e.g., C:)
        if len(normalized) >= 2 and normalized[1] == ':':
            return True

        # Split into path components and check each one
        # This is more reliable than string matching ".." which could match "file..txt"
        parts = Path(normalized).parts
        for part in parts:
            if part == "..":
                return True
            # Also check for null bytes which could cause issues
            if '\x00' in part:
                return True

        return False

    @staticmethod
    def _get_env_int(name: str, default: int) -> int:
        """Get integer value from environment variable with fallback to default."""
        try:
            return int(os.getenv(name, str(default)))
        except (ValueError, TypeError):
            return default

    @classmethod
    def _get_archive_limits(cls) -> tuple[int, int]:
        """Return per-file and total archive limits in bytes."""
        per_file_mb = cls._get_env_int("CHATBOOKS_MAX_FILE_IN_ARCHIVE_MB", 50)
        total_mb = cls._get_env_int("CHATBOOKS_MAX_UNCOMPRESSED_SIZE_MB", 500)
        if per_file_mb <= 0:
            per_file_mb = 50
        if total_mb <= 0:
            total_mb = 500
        return per_file_mb * 1024 * 1024, total_mb * 1024 * 1024

    @classmethod
    def _get_conversation_export_page_size(cls) -> int:
        """Return paging size for conversation message export."""
        size = cls._get_env_int("CHATBOOKS_CONVERSATION_EXPORT_PAGE_SIZE", 500)
        return size if size > 0 else 500

    @classmethod
    def _get_conversation_export_max_messages(cls) -> int | None:
        """Optional cap on exported messages per conversation (0 means unlimited)."""
        max_messages = cls._get_env_int("CHATBOOKS_CONVERSATION_EXPORT_MAX_MESSAGES", 0)
        return max_messages if max_messages > 0 else None

    @staticmethod
    def _get_max_message_image_bytes() -> int:
        """Return the maximum size for message images in bytes."""
        try:
            return int(core_settings.get("MAX_MESSAGE_IMAGE_BYTES", 5 * 1024 * 1024))
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
            return 5 * 1024 * 1024

    @classmethod
    def _get_export_retention_seconds(cls) -> int:
        """Return export retention duration in seconds (defaults to 24 hours)."""
        raw_hours = os.getenv("CHATBOOKS_EXPORT_RETENTION_DEFAULT_HOURS", "24")
        try:
            hours = int(raw_hours)
        except (TypeError, ValueError):
            hours = 24
        if hours <= 0:
            hours = 24
        return hours * 3600

    @classmethod
    def _get_download_ttl_seconds(cls) -> int:
        """Return download link TTL in seconds, bounded by export retention."""
        ttl = cls._get_env_int("CHATBOOKS_URL_TTL_SECONDS", 0)
        if ttl <= 0:
            ttl = cls._get_export_retention_seconds()
        return ttl

    @classmethod
    def _get_export_expiry(cls, now: datetime) -> datetime:
        """Compute export expiry timestamp from a reference time."""
        return now + timedelta(seconds=cls._get_export_retention_seconds())

    @classmethod
    def _get_download_expiry(cls, now: datetime, export_expires_at: datetime) -> datetime:
        """
        Compute download link expiry, capped by export expiry.

        Args:
            now: Current time used as the TTL anchor.
            export_expires_at: Timestamp when the export itself expires.

        Returns:
            Expiration timestamp for the download link.
        """
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if export_expires_at.tzinfo is None:
            export_expires_at = export_expires_at.replace(tzinfo=timezone.utc)
        ttl_seconds = cls._get_download_ttl_seconds()
        link_expires_at = now + timedelta(seconds=ttl_seconds)
        return link_expires_at if link_expires_at <= export_expires_at else export_expires_at

    @classmethod
    def _get_binary_limits_bytes(cls) -> dict[str, int]:
        """Parse per-type binary size limits from env JSON (MB -> bytes)."""
        raw = os.getenv("CHATBOOKS_BINARY_LIMITS_MB", "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("CHATBOOKS_BINARY_LIMITS_MB is not valid JSON; ignoring")
            return {}
        if not isinstance(parsed, dict):
            logger.warning("CHATBOOKS_BINARY_LIMITS_MB must be a JSON object; ignoring")
            return {}
        limits: dict[str, int] = {}
        for key, value in parsed.items():
            try:
                mb_value = float(value)
            except (TypeError, ValueError):
                continue
            if mb_value < 0:
                continue
            limits[str(key).strip().lower()] = int(mb_value * 1024 * 1024)
        return limits

    @staticmethod
    def _resolve_binary_limit(limits: dict[str, int], *keys: str) -> int | None:
        """Return the first matching size limit for the provided keys."""
        for key in keys:
            limit = limits.get(key)
            if limit is not None:
                return limit
        return None

    @staticmethod
    def _truthy_env(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return is_truthy(str(raw))

    @classmethod
    def _chat_dict_templates_enabled(cls) -> bool:
        """Return whether dictionary templating is globally enabled."""
        if cls._truthy_env("CHAT_DICT_TEMPLATES_ENABLED", False):
            return True
        try:
            cp = load_comprehensive_config()
            if cp and cp.has_section("Chat-Templating"):
                raw = cp.get("Chat-Templating", "enable_templates", fallback="false")
                return is_truthy(str(raw))
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
            pass
        return False

    @classmethod
    def _default_chatbook_template_metadata(cls) -> dict[str, Any]:
        """Build default manifest template metadata from environment settings."""
        mode = str(os.getenv("CHATBOOKS_TEMPLATE_MODE", "pass_through")).strip().lower()
        if mode not in _CHATBOOK_TEMPLATE_MODES:
            mode = "pass_through"

        template_defaults: dict[str, Any] = {}
        raw_defaults = os.getenv("CHATBOOKS_TEMPLATE_DEFAULTS_JSON", "").strip()
        if raw_defaults:
            try:
                parsed = json.loads(raw_defaults)
                if isinstance(parsed, dict):
                    template_defaults = parsed
            except json.JSONDecodeError:
                logger.warning("CHATBOOKS_TEMPLATE_DEFAULTS_JSON is invalid JSON; ignoring")

        timezone_value = str(
            os.getenv("CHATBOOKS_TEMPLATE_TIMEZONE")
            or os.getenv("TEMPLATE_DEFAULT_TZ")
            or "UTC"
        ).strip() or "UTC"
        locale_raw = os.getenv("CHATBOOKS_TEMPLATE_LOCALE") or os.getenv("TEMPLATE_DEFAULT_LOCALE")
        locale_value = str(locale_raw).strip() if locale_raw is not None else ""

        metadata: dict[str, Any] = {"template_mode": mode}
        if template_defaults:
            metadata["template_defaults"] = template_defaults
        if timezone_value:
            metadata["template_timezone"] = timezone_value
        if locale_value:
            metadata["template_locale"] = locale_value
        return metadata

    @staticmethod
    def _resolve_template_settings(manifest: ChatbookManifest) -> dict[str, Any]:
        metadata = dict((manifest.metadata or {}))
        mode = str(metadata.get("template_mode", "pass_through")).strip().lower()
        if mode not in _CHATBOOK_TEMPLATE_MODES:
            mode = "pass_through"

        defaults = metadata.get("template_defaults")
        if not isinstance(defaults, dict):
            defaults = {}

        timezone_value = str(
            metadata.get("template_timezone")
            or os.getenv("TEMPLATE_DEFAULT_TZ")
            or "UTC"
        ).strip() or "UTC"
        locale_raw = metadata.get("template_locale") or os.getenv("TEMPLATE_DEFAULT_LOCALE")
        locale_value = str(locale_raw).strip() if locale_raw is not None else ""

        return {
            "mode": mode,
            "defaults": defaults,
            "timezone": timezone_value,
            "locale": locale_value or None,
        }

    @staticmethod
    def _should_render_for_stage(template_mode: str, stage: str) -> bool:
        stage_norm = str(stage).strip().lower()
        mode_norm = str(template_mode).strip().lower()
        return (
            (stage_norm == "export" and mode_norm == "render_on_export")
            or (stage_norm == "import" and mode_norm == "render_on_import")
        )

    def _render_chatbook_text(
        self,
        text: Any,
        *,
        template_settings: dict[str, Any],
        stage: str,
        metrics_source: str = "chatbook",
        require_dict_templates_enabled: bool = False,
    ) -> Any:
        """Render text according to manifest template settings and stage."""
        if not isinstance(text, str) or "{{" not in text:
            return text
        if not self._should_render_for_stage(str(template_settings.get("mode", "pass_through")), stage):
            return text
        if require_dict_templates_enabled and not self._chat_dict_templates_enabled():
            return text

        env = TemplateEnv(
            timezone=str(template_settings.get("timezone") or "UTC"),
            locale=template_settings.get("locale"),
        )
        extra = dict(template_settings.get("defaults") or {})
        extra.setdefault("_metrics_source", metrics_source)
        ctx = TemplateContext(
            user={"id": self.user_id, "display_name": self.user_id},
            env=env,
            extra=extra,
        )
        return render_template(text, ctx, options_from_env())

    @staticmethod
    def _build_export_filename(name: str, timestamp: str) -> str:
        """Build a safe, length-limited export filename."""
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        if not safe_name:
            safe_name = "chatbook"
        suffix = f"_{timestamp}_{uuid4().hex[:8]}.zip"
        max_len = 255
        if len(safe_name) + len(suffix) > max_len:
            safe_name = safe_name[: max_len - len(suffix)]
            if not safe_name:
                safe_name = "chatbook"
        return f"{safe_name}{suffix}"

    def __init__(self, user_id: str | int, db: CharactersRAGDB, user_id_int: int | None = None):
        """
        Initialize the chatbook service for a specific user.

        Args:
            user_id: User identifier (string or integer)
            db: User's ChaChaNotes database instance
            user_id_int: Optional integer form of the user id for cross-database access
        """
        self.user_id_raw = user_id
        self.user_id = str(user_id)

        # Early validation: reject empty user_id to prevent security issues
        if not self.user_id or self.user_id.strip() == "":
            raise ValueError("user_id cannot be empty or whitespace-only")

        self.user_id_int: int | None = user_id_int
        if self.user_id_int is None:
            try:
                self.user_id_int = int(self.user_id)
            except (TypeError, ValueError):
                self.user_id_int = None
        self.db = db

        # Track TODOs once per session so we comply with PRD while exposing gaps
        self._todo_messages: set[str] = set()

        # In-process async task registry (best-effort cancellation)
        self._tasks: dict[str, asyncio.Task] = {}
        self._prompts_db: PromptsDatabase | None = None
        self._media_db: Any | None = None
        self._evaluations_db: EvaluationsDatabase | None = None
        self._chroma_manager: ChromaDBManager | None = None

        # Secure user-specific directory under the configured user DB base.
        user_id_value = self.user_id_int if self.user_id_int is not None else self.user_id
        self.user_data_dir = DatabasePaths.get_user_chatbooks_dir(user_id_value)
        self.export_dir = DatabasePaths.get_user_chatbooks_exports_dir(user_id_value)
        self.import_dir = DatabasePaths.get_user_chatbooks_imports_dir(user_id_value)
        self.temp_dir = DatabasePaths.get_user_chatbooks_temp_dir(user_id_value)
        for directory in (self.user_data_dir, self.export_dir, self.import_dir, self.temp_dir):
            try:
                directory.chmod(0o700)
            except OSError:
                logger.debug(f"Chatbooks: unable to set permissions on {directory}")

        # Jobs backend selection (core only)
        backend = (os.getenv("CHATBOOKS_JOBS_BACKEND") or os.getenv("TLDW_JOBS_BACKEND") or "").strip().lower()
        if backend and backend != "core":
            logger.warning("Chatbooks jobs backend override ignored; only core Jobs is supported now.")
        self._jobs_backend = "core"

        # Legacy Prompt Studio adapter placeholder (no longer used; core Jobs only)
        self._ps_job_adapter = None
        self._jobs_adapter = None
        self._jobs_db_path: Path | None = None
        try:
            from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
            self._jobs_db_path = ensure_jobs_tables()
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Jobs core backend migrations skipped: {exc}")
        try:
            from .jobs_adapter import ChatbooksJobsAdapter
            self._jobs_adapter = ChatbooksJobsAdapter(owner_user_id=self.user_id)
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Chatbooks: core Jobs adapter unavailable: {exc}")

        # Initialize job tracking tables
        self._init_job_tables()


    # -------------------------------------------------------------------------
    # Helper utilities (TODO markers ensure disparities with PRD are surfaced)
    # -------------------------------------------------------------------------
    def _note_todo(self, message: str) -> None:
        """Log a TODO item once to highlight parity gaps with the PRD."""
        if message not in self._todo_messages:
            logger.warning(f"TODO(chatbooks): {message}")
            self._todo_messages.add(message)

    def _resolve_import_archive_path(self, file_ref: str | Path) -> Path:
        """Resolve and validate a chatbook archive path within temp/imports directories."""
        ref = str(file_ref or "").strip()
        if not ref:
            raise ValidationError("Chatbook file path is required", field="file_path")

        base_dirs = [("import", self.import_dir.resolve()), ("temp", self.temp_dir.resolve())]
        base_map = dict(base_dirs)
        ref_path = Path(ref)
        base_hint: str | None = None

        if not (ref_path.is_absolute() or (ref_path.drive and ref_path.root)):
            token_parts = ref.split("/", 1)
            if token_parts[0] in base_map:
                base_hint = token_parts[0]
                if len(token_parts) == 1 or not token_parts[1]:
                    raise ValidationError("Chatbook file path is required", field="file_path")
                ref_path = Path(token_parts[1])

        bases_to_check = base_dirs
        if base_hint is not None:
            bases_to_check = [(base_hint, base_map[base_hint])]

        candidates: list[tuple[str, Path, Path]] = []
        if ref_path.is_absolute() or (ref_path.drive and ref_path.root):
            for base_name, base in bases_to_check:
                candidates.append((base_name, base, ref_path))
        else:
            for base_name, base in bases_to_check:
                candidates.append((base_name, base, base / ref_path))

        for _base_name, base, candidate in candidates:
            exists = False
            try:
                exists = candidate.exists()
            except OSError as exc:
                logger.debug("Chatbooks import: exists check failed for base {}: {}", _base_name, exc)
                exists = False
            if exists:
                try:
                    resolved = candidate.resolve(strict=True)
                except OSError as exc:
                    logger.debug("Chatbooks import: resolve(strict=True) failed for base {}: {}", _base_name, exc)
                    continue
                try:
                    resolved.relative_to(base)
                except ValueError:
                    continue
                return resolved
            try:
                resolved = candidate.resolve(strict=False)
            except OSError as exc:
                logger.debug("Chatbooks import: resolve(strict=False) failed for base {}: {}", _base_name, exc)
                continue
            try:
                resolved.relative_to(base)
            except ValueError:
                continue

        raise SecurityError(
            "Chatbook file path is outside allowed import directories",
            violation_type="import_path_outside_allowed_directories",
        )

    def _resolve_import_upload_path(self, file_ref: str | Path) -> Path:
        """Resolve an uploaded import file path within temp/import directories."""
        return self._resolve_import_archive_path(file_ref)

    def _build_import_file_token(self, resolved_path: Path) -> str:
        """Return a tokenized relative path for import job payloads."""
        base_dirs = [("import", self.import_dir.resolve()), ("temp", self.temp_dir.resolve())]
        for base_name, base in base_dirs:
            try:
                return f"{base_name}/{resolved_path.relative_to(base).as_posix()}"
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                continue
        logger.debug(
            "Chatbooks: path {} not under import/temp dirs; using filename only",
            resolved_path,
        )
        return resolved_path.name

    def _get_prompts_db(self) -> PromptsDatabase | None:
        """Lazily initialize and cache the prompts database."""
        if PromptsDatabase is None:
            self._note_todo("Prompts export/import requires PromptsDatabase module; skipping for current build.")
            return None
        if self._prompts_db is not None:
            return self._prompts_db
        if self.user_id_int is None:
            self._note_todo("Prompts export/import requires numeric user id to resolve database path.")
            return None
        try:
            db_path = DatabasePaths.get_prompts_db_path(self.user_id_int)
            self._prompts_db = PromptsDatabase(db_path, client_id=self.user_id)
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - defensive guard
            logger.warning(f"Failed to initialize PromptsDatabase for chatbooks export: {exc}")
            self._note_todo("Prompts export/import initialization failed; inspect logs for details.")
            self._prompts_db = None
        return self._prompts_db

    def _get_media_db(self) -> Any | None:
        """Lazily initialize and cache the media database."""
        if create_media_database is None:
            self._note_todo("Media export/import requires media DB factory support; skipping media coverage.")
            return None
        if self._media_db is not None:
            return self._media_db
        if self.user_id_int is None:
            self._note_todo("Media export/import requires numeric user id to resolve database path.")
            return None
        try:
            db_path = DatabasePaths.get_media_db_path(self.user_id_int)
            self._media_db = create_media_database(self.user_id, db_path=db_path)
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover
            logger.warning(f"Failed to initialize MediaDatabase for chatbooks export: {exc}")
            self._note_todo("Media export/import initialization failed; inspect logs for details.")
            self._media_db = None
        return self._media_db

    def _get_evaluations_db(self) -> EvaluationsDatabase | None:
        """Lazily initialize and cache the evaluations database."""
        if EvaluationsDatabase is None:
            self._note_todo("Evaluations export/import requires EvaluationsDatabase module; skipping evaluations coverage.")
            return None
        if self._evaluations_db is not None:
            return self._evaluations_db
        if self.user_id_int is None:
            self._note_todo("Evaluations export/import requires numeric user id to resolve database path.")
            return None
        try:
            db_path = DatabasePaths.get_evaluations_db_path(self.user_id_int)
            # EvaluationsDatabase handles backend resolution internally
            self._evaluations_db = EvaluationsDatabase(str(db_path))
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover
            logger.warning(f"Failed to initialize EvaluationsDatabase for chatbooks export: {exc}")
            self._note_todo("Evaluations export/import initialization failed; inspect logs for details.")
            self._evaluations_db = None
        return self._evaluations_db

    def _get_chroma_manager(self) -> ChromaDBManager | None:
        """Lazily initialize and cache the ChromaDB manager for embedding export."""
        if ChromaDBManager is None:
            self._note_todo("Embedding export requires ChromaDBManager; skipping.")
            return None
        if self._chroma_manager is not None:
            return self._chroma_manager
        try:
            cfg = core_settings.get("EMBEDDING_CONFIG", {}).copy()
            cfg["USER_DB_BASE_DIR"] = str(DatabasePaths.get_user_db_base_dir())
            self._chroma_manager = ChromaDBManager(
                user_id=self.user_id, user_embedding_config=cfg
            )
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"ChromaDB init failed for chatbooks: {exc}")
            self._chroma_manager = None
        return self._chroma_manager

    @staticmethod
    def _normalize_datetime(value: Any) -> Any:
        """Convert datetime-like values to ISO strings."""
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    @staticmethod
    def _convert_datetimes(obj: Any) -> Any:
        """Recursively convert datetime values to ISO 8601 strings."""
        if isinstance(obj, dict):
            return {k: ChatbookService._convert_datetimes(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ChatbookService._convert_datetimes(item) for item in obj]
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        """Robust timestamp parser for database rows."""
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            try:
                # Treat numeric input as Unix timestamp (UTC)
                # Bounds check: reject timestamps before 1970 or after year 9999
                # (approximately -86400 to 253402300800 seconds from epoch)
                MIN_TIMESTAMP = -86400  # Allow small negative for timezone edge cases
                MAX_TIMESTAMP = 253402300800  # Year 9999 approximately
                if value < MIN_TIMESTAMP or value > MAX_TIMESTAMP:
                    return None
                # Use fromtimestamp with timezone.utc, then strip tzinfo to get naive UTC datetime
                # (utcfromtimestamp is deprecated in Python 3.12+)
                return datetime.fromtimestamp(value, tz=timezone.utc).replace(tzinfo=None)
            except (OSError, OverflowError, ValueError):
                return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            # Support trailing Z (UTC)
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(text)
                return ChatbookService._normalize_timestamp_to_naive(parsed)
            except ValueError:
                pass
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(text, fmt)
                except ValueError:
                    continue
        return None

    @staticmethod
    def _normalize_timestamp_to_naive(value: datetime | None) -> datetime | None:
        """Convert aware timestamps to naive UTC for consistent downstream handling."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def _get_fallback_character_id(self) -> int | None:
        """Return a fallback character id (default assistant) if available."""
        try:
            record = self.db.get_character_card_by_id(1)
            if record and record.get("id"):
                return int(record["id"])
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            cursor = self.db.execute_query(
                "SELECT id FROM character_cards WHERE deleted = 0 ORDER BY id ASC LIMIT 1"
            )
            rows = self._fetch_results(cursor)
            if rows:
                row = rows[0]
                if isinstance(row, dict) and row.get("id") is not None:
                    return int(row["id"])
                if isinstance(row, (list, tuple)) and row:
                    return int(row[0])
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
            pass
        return None

    def _resolve_import_character_id(
        self,
        original_id: Any,
        character_id_map: dict[str, int] | None = None
    ) -> tuple[int | None, str | None]:
        """Resolve a character_id for imported conversations, falling back when needed."""
        if original_id is None or str(original_id).strip() == "":
            fallback = self._get_fallback_character_id()
            if fallback is None:
                return None, "Conversation missing character_id and no fallback character is available."
            return fallback, "Conversation missing character_id; using default character."

        key = str(original_id)
        if character_id_map and key in character_id_map:
            return character_id_map[key], None

        char_id_int: int | None = None
        try:
            char_id_int = int(original_id)
        except (TypeError, ValueError):
            char_id_int = None
        if char_id_int is not None:
            try:
                record = self.db.get_character_card_by_id(char_id_int)
                if record:
                    return char_id_int, None
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                pass

        fallback = self._get_fallback_character_id()
        if fallback is None:
            return None, f"Character {original_id} not found and no fallback character is available."
        return fallback, f"Character {original_id} not found; using default character."

    def _normalize_prompt_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Normalize prompt record for JSON export."""
        payload: dict[str, Any] = {}
        for key, value in record.items():
            payload[key] = self._normalize_datetime(value)
        return payload

    def _fetch_media_record(self, media_db: Any, identifier: str) -> dict[str, Any] | None:
        """Retrieve a media row by integer id or uuid."""
        record: dict[str, Any] | None = None
        try:
            record = get_media_by_id(media_db, int(identifier))
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
            record = None
        if not record:
            try:
                record = get_media_by_uuid(media_db, str(identifier))
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                record = None
        if record and isinstance(record, dict):
            return dict(record)
        return record

    def _normalize_media_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Normalize media row for JSON export."""
        payload: dict[str, Any] = {}
        for key, value in record.items():
            if key == "vector_embedding":
                # handled separately when include_embeddings is true
                continue
            if isinstance(value, (datetime,)):
                payload[key] = value.isoformat()
            elif isinstance(value, (bytes, bytearray, memoryview)):
                payload[key] = base64.b64encode(bytes(value)).decode("ascii")
            else:
                payload[key] = value
        return payload

    def _normalize_transcript_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Normalize transcript row from Media DB helpers."""
        payload: dict[str, Any] = {}
        for key, value in row.items():
            payload[key] = self._normalize_datetime(value)
        return payload

    def _normalize_evaluation_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Normalize evaluation definition for export."""
        payload: dict[str, Any] = {}
        for key, value in record.items():
            if key in {"eval_spec", "metadata"} and isinstance(value, str):
                try:
                    payload[key] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass
            payload[key] = self._normalize_datetime(value)
        return payload

    def _normalize_evaluation_run(self, run: dict[str, Any]) -> dict[str, Any]:
        """Normalize evaluation run for export."""
        payload: dict[str, Any] = {}
        for key, value in run.items():
            if key in {"config"} and isinstance(value, str):
                try:
                    payload[key] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass
            payload[key] = self._normalize_datetime(value)
        return payload

    def _get_conversation_messages_paged(self, conversation_id: str) -> tuple[list[dict[str, Any]], bool, int | None]:
        """Fetch all messages for a conversation using paging."""
        page_size = self._get_conversation_export_page_size()
        max_messages = self._get_conversation_export_max_messages()
        offset = 0
        messages: list[dict[str, Any]] = []
        truncated = False

        while True:
            batch = self.db.get_messages_for_conversation(
                conversation_id,
                limit=page_size,
                offset=offset,
            )
            if not batch:
                break

            if max_messages is not None:
                remaining = max_messages - len(messages)
                if remaining <= 0:
                    truncated = True
                    break
                if len(batch) > remaining:
                    messages.extend(batch[:remaining])
                    truncated = True
                    break

            messages.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size

        return messages, truncated, max_messages

    @staticmethod
    def _extension_from_mime(mime_type: str | None) -> str:
        """Infer a safe file extension for an attachment mime type."""
        if not mime_type:
            return ".bin"
        mapping = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/webp": ".webp",
            "image/gif": ".gif"
        }
        return mapping.get(mime_type.lower(), ".bin")

    def _fetch_results(self, cursor_or_list):
        """
        Helper to convert cursor or list to list of results.
        Handles both real database cursors and mocked list results.
        """
        if hasattr(cursor_or_list, 'fetchall'):
            # It's a cursor - fetch all results
            results = cursor_or_list.fetchall()
            if not results:
                return []

            # sqlite3.Row objects can be converted directly to dict
            # but we need to handle different cases
            results[0]

            # Try the simplest approach first - direct dict conversion
            try:
                # This works for sqlite3.Row objects
                return [dict(row) for row in results]
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                # If that fails, use cursor description
                if hasattr(cursor_or_list, 'description') and cursor_or_list.description:
                    columns = [desc[0] for desc in cursor_or_list.description]
                    return [dict(zip(columns, row)) for row in results]
                else:
                    # Can't convert to dict, return as tuples
                    return results
        else:
            # It's already a list (from mocked tests)
            return cursor_or_list

    def _get_conversation_by_name(self, name: str) -> dict[str, Any] | None:
        """Get conversation by name/title - wrapper for search method."""
        try:
            # First try FTS search
            if hasattr(self.db, 'search_conversations_by_title'):
                results = self.db.search_conversations_by_title(
                    name,
                    limit=10,
                    client_id=getattr(self.db, "client_id", None),
                )
                logger.debug(f"FTS search for conversation '{name}', found {len(results)} results")
                # Look for exact match
                for conv in results:
                    conv_title = conv.get('title')
                    conv_name = conv.get('name')
                    logger.debug(f"  Checking: title='{conv_title}', name='{conv_name}'")
                    if conv_title == name or conv_name == name:
                        logger.debug("  Found exact match via FTS!")
                        return conv

            # If FTS didn't find it, try direct query (FTS might not be updated yet)
            if hasattr(self.db, 'execute_query'):
                logger.debug(f"FTS failed, trying direct query for '{name}'")
                cursor = self.db.execute_query(
                    "SELECT * FROM conversations WHERE title = ? AND deleted = 0 LIMIT 1",
                    (name,)
                )
                # Fetch results from cursor
                if cursor:
                    results = cursor.fetchall() if hasattr(cursor, 'fetchall') else []
                    logger.debug(f"Direct query returned {len(results)} results")
                    if results and len(results) > 0:
                        logger.debug(f"Found conversation via direct query: {results[0]}")
                        # Convert tuple to dict if needed
                        if isinstance(results[0], tuple):
                            # Assume standard column order
                            return {'id': results[0][0], 'title': results[0][1] if len(results[0]) > 1 else name}
                        return results[0]
                else:
                    logger.debug("Direct query returned None/empty cursor")

            logger.debug(f"No match found for '{name}' via FTS or direct query")
            return None
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Error searching for conversation by name: {e}")
            return None

    def _get_note_by_title(self, title: str) -> dict[str, Any] | None:
        """Get note by title - wrapper for search method."""
        try:
            # First try FTS search
            if hasattr(self.db, 'search_notes'):
                results = self.db.search_notes(title, limit=10)
                logger.debug(f"FTS search for note '{title}', found {len(results)} results")
                # Look for exact match
                for note in results:
                    note_title = note.get('title')
                    logger.debug(f"  Checking note: title='{note_title}'")
                    if note_title == title:
                        logger.debug("  Found exact match via FTS!")
                        return note

            # If FTS didn't find it, try direct query (FTS might not be updated yet)
            if hasattr(self.db, 'execute_query'):
                logger.debug(f"FTS failed, trying direct query for note '{title}'")
                cursor = self.db.execute_query(
                    "SELECT * FROM notes WHERE title = ? AND deleted = 0 LIMIT 1",
                    (title,)
                )
                # Fetch results from cursor
                if cursor:
                    results = cursor.fetchall() if hasattr(cursor, 'fetchall') else []
                    logger.debug(f"Direct query returned {len(results)} results for note")
                    if results and len(results) > 0:
                        logger.debug(f"Found note via direct query: {results[0]}")
                        # Convert tuple to dict if needed
                        if isinstance(results[0], tuple):
                            # Assume standard column order
                            return {'id': results[0][0], 'title': results[0][1] if len(results[0]) > 1 else title}
                        return results[0]
                else:
                    logger.debug("Direct query returned None/empty cursor for note")

            logger.debug(f"No match found for note '{title}' via FTS or direct query")
            return None
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Error searching for note by title: {e}")
            return None

    def _init_job_tables(self):
        """Initialize database tables for job tracking."""
        try:
            # Export jobs table
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS export_jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chatbook_name TEXT NOT NULL,
                    output_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    progress_percentage INTEGER DEFAULT 0,
                    total_items INTEGER DEFAULT 0,
                    processed_items INTEGER DEFAULT 0,
                    file_size_bytes INTEGER,
                    download_url TEXT,
                    expires_at TIMESTAMP
                )
            """)

            # Import jobs table
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS import_jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chatbook_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    progress_percentage INTEGER DEFAULT 0,
                    total_items INTEGER DEFAULT 0,
                    processed_items INTEGER DEFAULT 0,
                    successful_items INTEGER DEFAULT 0,
                    failed_items INTEGER DEFAULT 0,
                    skipped_items INTEGER DEFAULT 0,
                    conflicts TEXT,  -- JSON array
                    warnings TEXT    -- JSON array
                )
            """)
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error initializing job tables: {e}")

    # Alias for compatibility with tests
    async def export_chatbook(self, **kwargs):
        """Alias for create_chatbook to match test expectations."""
        # Extract user_id for internal use but don't pass it to create_chatbook
        kwargs.pop('user_id', None)

        # Extract chatbook_name and use it as 'name'
        if 'chatbook_name' in kwargs:
            kwargs['name'] = kwargs.pop('chatbook_name')

        # Extract options and merge them into kwargs
        if 'options' in kwargs:
            options = kwargs.pop('options')
            kwargs.update(options)

        # Map content_types to content_selections for compatibility
        if 'content_types' in kwargs:
            content_types = kwargs.pop('content_types')
            # Convert simple list to dict format
            content_selections = {}
            for ct in content_types:
                if ct == "conversations":
                    # Get all conversation IDs when none specified
                    conv_ids = []
                    try:
                        cursor = self.db.execute_query(
                            "SELECT id FROM conversations WHERE deleted = 0"
                        )
                        rows = self._fetch_results(cursor)
                        conv_ids = [
                            (row.get("id") if isinstance(row, dict) else row[0])
                            for row in (rows or [])
                            if row is not None
                        ]
                    except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(f"Error getting conversations for export: {e}")
                    content_selections[ContentType.CONVERSATION] = conv_ids
                elif ct == "characters":
                    # Get all character IDs when none specified
                    char_ids = []
                    try:
                        cursor = self.db.execute_query(
                            "SELECT id FROM character_cards WHERE deleted = 0"
                        )
                        rows = self._fetch_results(cursor)
                        char_ids = [
                            str(row.get("id") if isinstance(row, dict) else row[0])
                            for row in (rows or [])
                            if row is not None
                        ]
                    except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(f"Error getting characters for export: {e}")
                    content_selections[ContentType.CHARACTER] = char_ids
                elif ct == "notes":
                    # Get all note IDs when none specified
                    note_ids = []
                    try:
                        cursor = self.db.execute_query(
                            "SELECT id FROM notes WHERE deleted = 0"
                        )
                        rows = self._fetch_results(cursor)
                        note_ids = [
                            (row.get("id") if isinstance(row, dict) else row[0])
                            for row in (rows or [])
                            if row is not None
                        ]
                    except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(f"Error getting notes for export: {e}")
                    content_selections[ContentType.NOTE] = note_ids
                elif ct == "world_books":
                    # Get all world book IDs when none specified
                    wb_ids = []
                    try:
                        if hasattr(self, 'world_books') and self.world_books:
                            world_books = self.world_books.list_world_books()
                            wb_ids = [str(wb['id']) for wb in world_books] if world_books else []
                        else:
                            # Fallback to direct database query
                            logger.debug("WorldBookService not available, using direct query")
                    except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(f"Error getting world books for export: {e}")
                    content_selections[ContentType.WORLD_BOOK] = wb_ids
                elif ct == "dictionaries":
                    # Get all dictionary IDs when none specified
                    dict_ids = []
                    try:
                        if hasattr(self, 'dictionaries') and self.dictionaries:
                            dictionaries = self.dictionaries.list_dictionaries()
                            dict_ids = [str(d['id']) for d in dictionaries] if dictionaries else []
                        else:
                            # Fallback to direct database query
                            logger.debug("ChatDictionary not available, using direct query")
                    except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(f"Error getting dictionaries for export: {e}")
                    content_selections[ContentType.DICTIONARY] = dict_ids
            kwargs['content_selections'] = content_selections

        # Set default values for required params if missing
        kwargs.setdefault('name', 'Test Export')
        kwargs.setdefault('description', 'Test Description')

        # Handle async_job parameter
        if 'async_job' in kwargs:
            kwargs['async_mode'] = kwargs.pop('async_job')

        result = await self.create_chatbook(**kwargs)

        # Convert tuple result to dict for tests
        if isinstance(result, tuple):
            success = result[0]
            message = result[1] if len(result) > 1 else ""
            payload = result[2] if len(result) > 2 else None
            is_async = bool(kwargs.get('async_mode'))
            file_path = None if is_async else payload
            job_id = payload if is_async else None
            content_summary: dict[str, int] = {}

            # If we have a file path (sync export), read manifest to populate summary
            if file_path:
                try:
                    from zipfile import ZipFile
                    with ZipFile(file_path, 'r') as zf:
                        if 'manifest.json' in zf.namelist():
                            import json as _json
                            manifest_data = _json.loads(zf.read('manifest.json'))
                            # Pull totals from statistics (fallback to top-level for legacy manifests)
                            stats = manifest_data.get('statistics', {}) or {}
                            totals = {
                                'conversations': stats.get('total_conversations', manifest_data.get('total_conversations', 0)),
                                'notes': stats.get('total_notes', manifest_data.get('total_notes', 0)),
                                'characters': stats.get('total_characters', manifest_data.get('total_characters', 0)),
                                'world_books': stats.get('total_world_books', manifest_data.get('total_world_books', 0)),
                                'dictionaries': stats.get('total_dictionaries', manifest_data.get('total_dictionaries', 0)),
                                'documents': stats.get('total_documents', manifest_data.get('total_documents', 0)),
                            }
                            # Only include non-zero entries to keep it tidy
                            content_summary = {k: v for k, v in totals.items() if isinstance(v, int) and v >= 0}
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as _e:
                    # Fallback to empty summary on any error
                    logger.debug(f"Could not read manifest for content summary: {_e}")

            return {
                "success": success,
                "message": message,
                "file_path": file_path,
                "job_id": job_id,
                "status": "pending" if is_async else "completed",
                "content_summary": content_summary,
            }
        return result

    async def create_chatbook(
        self,
        name: str,
        description: str,
        content_selections: dict[ContentType, list[str]],
        author: str | None = None,
        include_media: bool = False,
        media_quality: str = "compressed",
        include_embeddings: bool = False,
        include_generated_content: bool = True,
        tags: list[str] | None = None,
        categories: list[str] | None = None,
        async_mode: bool = False,
        request_id: str | None = None
    ) -> tuple[bool, str, str | None]:
        """
        Create a chatbook from selected content.

        Args:
            name: Chatbook name
            description: Chatbook description
            content_selections: Content to include by type and IDs
            author: Author name
            include_media: Include media files
            media_quality: Media quality level
            include_embeddings: Include embeddings
            include_generated_content: Include generated documents
            tags: Chatbook tags
            categories: Chatbook categories
            async_mode: Run as background job

        Returns:
            Tuple of (success, message, job_id or file_path)
        """
        if async_mode:
            # Create job and run asynchronously
            # If using Prompt Studio backend, create PS job first and reuse its id
            job_id = None
            if self._jobs_backend == "prompt_studio" and self._ps_job_adapter is not None:
                payload = {
                    "domain": "chatbooks",
                    "job_type": "export",
                    "user_id": self.user_id,
                    "name": name,
                    "include_media": include_media,
                    "include_embeddings": include_embeddings,
                    "include_generated_content": include_generated_content,
                    "tags": tags or [],
                    "categories": categories or [],
                }
                try:
                    ps_job = self._ps_job_adapter.create_export_job(payload, request_id=request_id)
                    if ps_job and ps_job.get("id") is not None:
                        job_id = str(ps_job["id"])  # mirror PS id
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                    logger.warning(f"Failed to create PS export job, falling back to core: {e}")
                    job_id = None
            if job_id is None:
                job_id = str(uuid4())
            job = ExportJob(
                job_id=job_id,
                user_id=self.user_id,
                status=ExportStatus.PENDING,
                chatbook_name=name
            )

            # Store job in database
            self._save_export_job(job)

            # Start async processing depending on backend
            if self._jobs_backend == "core":
                # Enqueue into core Jobs and start worker if needed
                job_created = None
                enqueue_error: str | None = None
                try:
                    from tldw_Server_API.app.core.Jobs.manager import JobManager
                    if not hasattr(self, "_core_jobs"):
                        self._core_jobs = JobManager()
                    payload = {
                        "action": "export",
                        "chatbooks_job_id": job_id,
                        "name": name,
                        "description": description,
                        "content_selections": {k.value if hasattr(k, 'value') else str(k): v for k, v in content_selections.items()},
                        "author": author,
                        "include_media": include_media,
                        "media_quality": media_quality,
                        "include_embeddings": include_embeddings,
                        "include_generated_content": include_generated_content,
                        "tags": tags or [],
                        "categories": categories or [],
                    }
                    job_created = self._core_jobs.create_job(
                        domain="chatbooks",
                        queue="default",
                        job_type="export",
                        payload=payload,
                        owner_user_id=self.user_id,
                        priority=5,
                        max_retries=3,
                        request_id=request_id,
                    )
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                    enqueue_error = str(e)
                    logger.warning(f"Failed to enqueue export job into core Jobs: {e}")
                if not job_created:
                    err_msg = enqueue_error or "Failed to enqueue export job"
                    job.status = ExportStatus.FAILED
                    job.completed_at = datetime.now(timezone.utc)
                    job.error_message = err_msg
                    try:
                        self._save_export_job(job)
                    except _CHATBOOK_NONCRITICAL_EXCEPTIONS as save_err:
                        logger.warning(f"Failed to persist failed export job state: {save_err}")
                    return False, f"Export job failed to enqueue: {err_msg}", job_id
            elif self._jobs_backend == "prompt_studio":
                # Do not start local processing when using Prompt Studio backend.
                # PS worker (external) is responsible for running the job.
                pass

            return True, f"Export job started: {job_id}", job_id
        else:
            # Run synchronously (wrapped in async)
            return await self._create_chatbook_sync_wrapper(
                name, description, content_selections,
                author, include_media, media_quality, include_embeddings,
                include_generated_content, tags, categories
            )

    def _with_transaction(self, func, *args, **kwargs):
        """Execute a function within a database transaction."""
        conn = None
        try:
            # Get connection and start transaction
            conn = self.db.get_connection() if hasattr(self.db, 'get_connection') else None
            if conn:
                conn.execute("BEGIN TRANSACTION")

            # Execute the function
            result = func(*args, **kwargs)

            # Commit if we have a connection
            if conn:
                conn.execute("COMMIT")

            return result

        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            # Rollback on error
            if conn:
                try:
                    conn.execute("ROLLBACK")
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e2:
                    logger.warning(f"Transaction rollback failed: error={e2}")
            logger.error(f"Transaction rolled back: {e}")
            raise
        finally:
            # Close connection
            if conn:
                try:
                    conn.close()
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e3:
                    logger.warning(f"Connection close failed after transaction: error={e3}")

    async def continue_chatbook_export(
        self,
        export_id: str,
        continuations: list[dict[str, Any]],
        name: str | None = None,
        async_mode: bool = False,
        request_id: str | None = None,
    ) -> tuple[bool, str, str | None]:
        """
        Continue a truncated export by producing a linked chatbook with continuation data.

        Args:
            export_id: Original export's export_id from manifest.
            continuations: Continuation tokens from the original manifest's truncation metadata.
            name: Override name for the continuation chatbook.
            async_mode: Whether to run asynchronously (not yet supported for continuation).
            request_id: Optional request ID for tracing.

        Returns:
            Tuple of (success, message, file_path).
        """
        if async_mode:
            return False, "Async continuation exports are not yet supported", None

        work_dir: Path | None = None
        output_path: Path | None = None
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            work_dir = self.temp_dir / f"continue_{timestamp}_{uuid4().hex[:8]}"
            work_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

            # Determine sequence number from export_id, keeping the base stable
            base_id = export_id.split("_cont_")[0]
            seq = 1
            if "_cont_" in export_id:
                try:
                    seq = int(export_id.rsplit("_cont_", 1)[-1]) + 1
                except (TypeError, ValueError):
                    seq = 1
            cont_export_id = f"{base_id}_cont_{seq}"

            cont_name = name or f"Continuation of {export_id}"
            manifest = ChatbookManifest(
                version=ChatbookVersion.V1,
                name=cont_name,
                description=f"Continuation export linked to {export_id}",
                user_id=hashlib.sha256(self.user_id.encode()).hexdigest()[:16],
                export_id=cont_export_id,
                metadata={"continues_export_id": export_id},
            )
            manifest.binary_limits = self._get_binary_limits_bytes()
            content = ChatbookContent()

            evals_db = self._get_evaluations_db()

            raw_max_rows = os.getenv("CHATBOOKS_EVAL_EXPORT_MAX_ROWS", "200")
            try:
                max_rows = int(raw_max_rows)
            except (TypeError, ValueError):
                max_rows = 200

            eval_dir = work_dir / "content" / "evaluations"
            eval_dir.mkdir(parents=True, exist_ok=True)

            for token in continuations:
                eval_id = token.get("evaluation_id")
                after = token.get("continuation_token")
                if not eval_id or not after:
                    logger.debug(f"Skipping invalid continuation token: {token}")
                    continue
                if evals_db is None:
                    self._note_todo("Continuation requires EvaluationsDatabase; skipping.")
                    continue

                try:
                    runs, has_more = evals_db.list_runs(
                        eval_id=str(eval_id), limit=max_rows,
                        after=str(after), return_has_more=True
                    )
                    runs_payload = [self._normalize_evaluation_run(run) for run in runs]

                    eval_data: dict[str, Any] = {
                        "evaluation_id": str(eval_id),
                        "continued_from": str(after),
                        "runs": runs_payload,
                    }

                    if has_more:
                        eval_data["truncated"] = True
                        eval_data["max_rows"] = max_rows
                        truncation = manifest.truncation.setdefault("evaluations", {})
                        truncation["truncated"] = True
                        truncation["max_rows"] = max_rows
                        truncation["exported_count"] = truncation.get("exported_count", 0) + len(runs_payload)
                        if runs_payload:
                            last_run_id = runs_payload[-1].get("id")
                            if last_run_id:
                                new_continuations = truncation.setdefault("continuations", [])
                                new_continuations.append({
                                    "evaluation_id": str(eval_id),
                                    "run_id": str(last_run_id),
                                    "continuation_token": str(last_run_id)
                                })

                    eval_file = eval_dir / f"evaluation_{eval_id}_cont.json"
                    with open(eval_file, "w", encoding="utf-8") as ef:
                        json.dump(eval_data, ef, indent=2, ensure_ascii=False)
                    content.evaluations[str(eval_id)] = eval_data
                    manifest.content_items.append(ContentItem(
                        id=str(eval_id),
                        type=ContentType.EVALUATION,
                        title=f"Evaluation {eval_id} (continuation)",
                        file_path=f"content/evaluations/{eval_file.name}"
                    ))
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(f"Failed to continue evaluation {eval_id}: {exc}")

            manifest.total_evaluations = len(content.evaluations)

            # Write manifest
            manifest_path = work_dir / "manifest.json"
            async with aiofiles.open(manifest_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))

            # Create archive
            output_filename = self._build_export_filename(cont_name, timestamp)
            output_path = self.export_dir / output_filename
            await self._create_zip_archive_async(work_dir, output_path)

            manifest.total_size_bytes = output_path.stat().st_size
            async with aiofiles.open(manifest_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
            await self._create_zip_archive_async(work_dir, output_path)

            return True, "Continuation chatbook created successfully", str(output_path)

        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error creating continuation chatbook: {e}")
            if output_path and output_path.exists():
                try:
                    await asyncio.to_thread(output_path.unlink)
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                    pass
            return False, f"Error creating continuation chatbook: {e}", None
        finally:
            if work_dir and work_dir.exists():
                try:
                    await asyncio.to_thread(shutil.rmtree, work_dir)
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                    pass

    async def _create_chatbook_sync_wrapper(
        self,
        name: str,
        description: str,
        content_selections: dict[ContentType, list[str]],
        author: str | None = None,
        include_media: bool = False,
        media_quality: str = "compressed",
        include_embeddings: bool = False,
        include_generated_content: bool = True,
        tags: list[str] | None = None,
        categories: list[str] | None = None
    ) -> tuple[bool, str, str | None]:
        """
        Wrapper for synchronous chatbook creation.

        Returns:
            Tuple of (success, message, file_path)
        """
        work_dir: Path | None = None
        output_path: Path | None = None
        try:
            # Create working directory in secure temp location
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            work_dir = self.temp_dir / f"export_{timestamp}_{uuid4().hex[:8]}"
            work_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

            # Initialize manifest
            manifest = ChatbookManifest(
                version=ChatbookVersion.V1,
                name=name,
                description=description,
                author=author,
                user_id=hashlib.sha256(self.user_id.encode()).hexdigest()[:16],  # Anonymized
                include_media=include_media,
                include_embeddings=include_embeddings,
                include_generated_content=include_generated_content,
                media_quality=media_quality,
                tags=tags or [],
                categories=categories or [],
                export_id=str(uuid4()),
                metadata=self._default_chatbook_template_metadata(),
            )
            manifest.binary_limits = self._get_binary_limits_bytes()

            # Collect content
            content = ChatbookContent()

            # Process each content type
            if ContentType.CONVERSATION in content_selections:
                self._collect_conversations(
                    content_selections[ContentType.CONVERSATION],
                    work_dir, manifest, content
                )

            if ContentType.NOTE in content_selections:
                self._collect_notes(
                    content_selections[ContentType.NOTE],
                    work_dir, manifest, content
                )

            if ContentType.CHARACTER in content_selections:
                self._collect_characters(
                    content_selections[ContentType.CHARACTER],
                    work_dir, manifest, content
                )

            if ContentType.WORLD_BOOK in content_selections:
                self._collect_world_books(
                    content_selections[ContentType.WORLD_BOOK],
                    work_dir, manifest, content
                )

            if ContentType.DICTIONARY in content_selections:
                self._collect_dictionaries(
                    content_selections[ContentType.DICTIONARY],
                    work_dir, manifest, content
                )

            if ContentType.MEDIA in content_selections:
                self._collect_media_items(
                    content_selections[ContentType.MEDIA],
                    work_dir, manifest, content,
                    include_media=include_media,
                    include_embeddings=include_embeddings
                )

            if ContentType.PROMPT in content_selections:
                self._collect_prompts(
                    content_selections[ContentType.PROMPT],
                    work_dir, manifest, content
                )

            if ContentType.EVALUATION in content_selections:
                self._collect_evaluations(
                    content_selections[ContentType.EVALUATION],
                    work_dir, manifest, content
                )

            if ContentType.EMBEDDING in content_selections:
                self._collect_embeddings(
                    content_selections[ContentType.EMBEDDING],
                    work_dir, manifest, content
                )

            if include_generated_content and ContentType.GENERATED_DOCUMENT in content_selections:
                self._collect_generated_documents(
                    content_selections[ContentType.GENERATED_DOCUMENT],
                    work_dir, manifest, content
                )

            # Update statistics
            manifest.total_conversations = len(content.conversations)
            manifest.total_notes = len(content.notes)
            manifest.total_characters = len(content.characters)
            manifest.total_media_items = len(content.media)
            manifest.total_prompts = len(content.prompts)
            manifest.total_evaluations = len(content.evaluations)
            manifest.total_embeddings = len(content.embeddings)
            manifest.total_world_books = len(content.world_books)
            manifest.total_dictionaries = len(content.dictionaries)
            manifest.total_documents = len(content.generated_documents)

            # Write manifest asynchronously
            manifest_path = work_dir / "manifest.json"
            async def _write_manifest() -> None:
                """Write the current manifest to disk as formatted JSON."""
                async with aiofiles.open(manifest_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
            await _write_manifest()

            # Create README asynchronously
            await self._create_readme_async(work_dir, manifest)

            # Create archive in secure export directory
            output_filename = self._build_export_filename(name, timestamp)
            output_path = self.export_dir / output_filename
            await self._create_zip_archive_async(work_dir, output_path)

            # Update manifest with final archive size; re-zip if manifest changes size.
            for _ in range(10):
                archive_size = output_path.stat().st_size
                if manifest.total_size_bytes == archive_size:
                    break
                manifest.total_size_bytes = archive_size
                await _write_manifest()
                await self._create_zip_archive_async(work_dir, output_path)

            # Store file path in job record (will be retrieved by job_id)
            # No direct filename access for security

            return True, "Chatbook created successfully", str(output_path)

        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error creating chatbook: {e}")
            if output_path and output_path.exists():
                try:
                    await asyncio.to_thread(output_path.unlink)
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as cleanup_err:
                    logger.warning(f"Failed to remove partial archive {output_path}: {cleanup_err}")
            return False, f"Error creating chatbook: {str(e)}", None
        finally:
            if work_dir and work_dir.exists():
                try:
                    await asyncio.to_thread(shutil.rmtree, work_dir)
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as cleanup_err:
                    logger.warning(f"Failed to remove work directory {work_dir}: {cleanup_err}")

    async def _create_chatbook_job_async(
        self,
        job_id: str,
        name: str,
        description: str,
        content_selections: dict[ContentType, list[str]],
        author: str | None,
        include_media: bool,
        media_quality: str,
        include_embeddings: bool,
        include_generated_content: bool,
        tags: list[str],
        categories: list[str]
    ):
        """
        Asynchronously create a chatbook with job tracking.
        """
        # Get job from database
        job = self._get_export_job(job_id)
        if not job:
            return

        try:
            # Update job status
            job.status = ExportStatus.IN_PROGRESS
            job.started_at = datetime.now(timezone.utc)
            self._save_export_job(job)
            # PS backend: reflect processing
            if getattr(self, "_jobs_backend", "core") == "prompt_studio" and getattr(self, "_ps_job_adapter", None) is not None:
                try:
                    self._ps_job_adapter.update_status(int(job.job_id), "in_progress")
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"PS adapter update_status failed for export job {job.job_id}: {e}")

            # Create chatbook using the sync wrapper (could be made truly async)
            success, message, file_path = await self._create_chatbook_sync_wrapper(
                name, description, content_selections,
                author, include_media, media_quality, include_embeddings,
                include_generated_content, tags, categories
            )

            if success:
                # Update job with success; respect cancellation
                latest = self._get_export_job(job.job_id)
                if latest and latest.status == ExportStatus.CANCELLED:
                    # PS backend: reflect cancellation terminal state
                    if getattr(self, "_jobs_backend", "core") == "prompt_studio" and getattr(self, "_ps_job_adapter", None) is not None:
                        try:
                            self._ps_job_adapter.update_status(int(job.job_id), "cancelled")
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                            logger.debug(f"PS adapter update_status (cancelled) failed for export job {job.job_id}: {e}")
                    return
                job.status = ExportStatus.COMPLETED
                now_utc = datetime.now(timezone.utc)
                job.completed_at = now_utc
                job.output_path = file_path
                job.file_size_bytes = Path(file_path).stat().st_size if file_path else None
                # Build (optionally signed) download URL and expiry
                job.expires_at = self._get_export_expiry(now_utc)
                download_expires_at = self._get_download_expiry(now_utc, job.expires_at)
                job.download_url = self._build_download_url(job.job_id, download_expires_at)
            else:
                # Update job with failure
                job.status = ExportStatus.FAILED
                job.completed_at = datetime.now(timezone.utc)
                job.error_message = message
            # PS backend: reflect terminal state
            if getattr(self, "_jobs_backend", "core") == "prompt_studio" and getattr(self, "_ps_job_adapter", None) is not None:
                try:
                    if job.status == ExportStatus.COMPLETED:
                        self._ps_job_adapter.update_status(int(job.job_id), "completed", result={"path": job.output_path})
                    elif job.status == ExportStatus.FAILED:
                        self._ps_job_adapter.update_status(int(job.job_id), "failed", error_message=job.error_message)
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"PS adapter update_status (completion) failed for export job {job.job_id}: {e}")
            self._save_export_job(job)

        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            # Update job with error
            job.status = ExportStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error_message = str(e)
            self._save_export_job(job)
            if getattr(self, "_jobs_backend", "core") == "prompt_studio" and getattr(self, "_ps_job_adapter", None) is not None:
                try:
                    self._ps_job_adapter.update_status(int(job.job_id), "failed", error_message=str(e))
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as ps_err:
                    logger.debug(f"PS adapter update_status (failed) failed for export job {job.job_id}: {ps_err}")

    async def import_chatbook(
        self,
        file_path: str,
        content_selections: dict[ContentType, list[str]] | None = None,
        conflict_resolution: ConflictResolution | str | None = None,
        conflict_strategy: str | None = None,  # Alias for conflict_resolution (for test compatibility)
        prefix_imported: bool = False,
        import_media: bool = False,
        import_embeddings: bool = False,
        async_mode: bool = False,
        request_id: str | None = None,
        source_format: str = "chatbook",
        selected_openwebui_user_id: str | None = None,
    ) -> tuple[bool, str, str | dict[str, Any] | None]:
        """
        Import a chatbook.

        Args:
            file_path: Path to chatbook file
            content_selections: Specific content to import
            conflict_resolution: How to handle conflicts
            prefix_imported: Add prefix to imported items
            import_media: Import media files (not supported yet)
            import_embeddings: Import embeddings (not supported yet)
            async_mode: Run as background job
            selected_openwebui_user_id: Selected OpenWebUI source user id for DB imports

        Returns:
            Tuple of (success, message, result) where result is:
            - job_id (str) if async_mode=True
            - sync import result dict if async_mode=False
        """
        # Handle both conflict_resolution and conflict_strategy (for test compatibility)
        if conflict_strategy and not conflict_resolution:
            conflict_resolution = conflict_strategy

        # Convert string to enum if needed
        if isinstance(conflict_resolution, str):
            try:
                conflict_resolution = ConflictResolution(conflict_resolution)
            except (ValueError, KeyError):
                # Log and default to skip if invalid value provided
                logger.warning(
                    f"Invalid conflict_resolution value '{conflict_resolution}', "
                    f"defaulting to 'skip'. Valid values: {[e.value for e in ConflictResolution]}"
                )
                conflict_resolution = ConflictResolution.SKIP
        elif conflict_resolution is None:
            # Default to skip if not specified
            conflict_resolution = ConflictResolution.SKIP

        # Reject unsupported conflict strategies until implemented
        unsupported_conflicts = {ConflictResolution.OVERWRITE, ConflictResolution.MERGE, ConflictResolution.ASK}
        if conflict_resolution in unsupported_conflicts:
            return False, (
                f"Conflict resolution '{conflict_resolution.value}' is not supported yet. "
                "Use 'skip' or 'rename'."
            ), None

        # Reject media/embedding imports until implemented
        if import_media or import_embeddings:
            return False, (
                "Media/embedding imports are not supported yet. "
                "Set import_media=false and import_embeddings=false."
            ), None

        source_format_value = getattr(source_format, "value", str(source_format or "chatbook")).strip().lower()
        if source_format_value not in {"chatbook", "openwebui_json", "openwebui_db"}:
            return False, f"Unsupported import source format: {source_format}", None

        if source_format_value in {"openwebui_json", "openwebui_db"}:
            if content_selections:
                return False, "OpenWebUI imports do not support content selections; all valid chats are imported.", None
            if import_media or import_embeddings:
                return False, "OpenWebUI imports do not support media or embedding import options.", None
        if source_format_value == "openwebui_db" and not (selected_openwebui_user_id or "").strip():
            return False, "selected_openwebui_user_id is required for OpenWebUI DB imports", None

        # Reject explicit requests for unsupported content types
        if content_selections:
            unsupported_types = {
                ContentType.MEDIA,
                ContentType.EMBEDDING,
                ContentType.PROMPT,
                ContentType.EVALUATION,
                ContentType.GENERATED_DOCUMENT,
            }
            requested = [
                ct.value if hasattr(ct, "value") else str(ct)
                for ct in content_selections
                if ct in unsupported_types
            ]
            if requested:
                return False, (
                    "Import for content types is not supported yet: "
                    + ", ".join(sorted(set(requested)))
                ), None

        try:
            if source_format_value in {"openwebui_json", "openwebui_db"}:
                resolved_path = self._resolve_import_upload_path(file_path)
            else:
                resolved_path = self._resolve_import_archive_path(file_path)
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"Chatbooks import rejected file path: {exc}")
            detail = (
                "Invalid or potentially malicious import file"
                if source_format_value == "openwebui_json"
                else "Invalid or potentially malicious archive file"
            )
            return False, detail, None
        file_token = self._build_import_file_token(resolved_path)

        if async_mode:
            # Create job and run asynchronously
            job_id = None
            if getattr(self, "_jobs_backend", "core") == "prompt_studio" and getattr(self, "_ps_job_adapter", None) is not None:
                payload = {
                    "domain": "chatbooks",
                    "job_type": "import",
                    "user_id": self.user_id,
                    "path": file_token,
                    "file_token": file_token,
                    "source_format": source_format_value,
                    "selected_openwebui_user_id": selected_openwebui_user_id,
                    "import_media": import_media,
                    "import_embeddings": import_embeddings,
                    "conflict_resolution": str(conflict_resolution.value if hasattr(conflict_resolution, 'value') else conflict_resolution),
                }
                try:
                    ps_job = self._ps_job_adapter.create_import_job(payload, request_id=request_id)
                    if ps_job and ps_job.get("id") is not None:
                        job_id = str(ps_job["id"])  # mirror PS id
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                    logger.warning(f"Failed to create PS import job, falling back to core: {e}")
                    job_id = None
            if job_id is None:
                job_id = str(uuid4())
            job = ImportJob(
                job_id=job_id,
                user_id=self.user_id,
                status=ImportStatus.PENDING,
                chatbook_path=file_token
            )

            # Store job in database
            self._save_import_job(job)

            # Start async task
            if self._jobs_backend == "core":
                job_created = None
                enqueue_error: str | None = None
                try:
                    from tldw_Server_API.app.core.Jobs.manager import JobManager
                    if not hasattr(self, "_core_jobs"):
                        self._core_jobs = JobManager()
                    payload = {
                        "action": "import",
                        "chatbooks_job_id": job_id,
                        "file_token": file_token,
                        "source_format": source_format_value,
                        "selected_openwebui_user_id": selected_openwebui_user_id,
                        "content_selections": {k.value if hasattr(k, 'value') else str(k): v for k, v in (content_selections or {}).items()},
                        "conflict_resolution": conflict_resolution.value if hasattr(conflict_resolution, 'value') else str(conflict_resolution),
                        "prefix_imported": bool(prefix_imported),
                        "import_media": bool(import_media),
                        "import_embeddings": bool(import_embeddings),
                    }
                    job_created = self._core_jobs.create_job(
                        domain="chatbooks",
                        queue="default",
                        job_type="import",
                        payload=payload,
                        owner_user_id=self.user_id,
                        priority=5,
                        max_retries=3,
                        request_id=request_id,
                    )
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                    enqueue_error = str(e)
                    logger.warning(f"Failed to enqueue import job into core Jobs: {e}")
                if not job_created:
                    err_msg = enqueue_error or "Failed to enqueue import job"
                    job.status = ImportStatus.FAILED
                    job.completed_at = datetime.now(timezone.utc)
                    job.error_message = err_msg
                    try:
                        self._save_import_job(job)
                    except _CHATBOOK_NONCRITICAL_EXCEPTIONS as save_err:
                        logger.warning(f"Failed to persist failed import job state: {save_err}")
                    return False, f"Import job failed to enqueue: {err_msg}", job_id
            else:
                task = asyncio.create_task(self._import_chatbook_async(
                    job_id, str(resolved_path), content_selections,
                    conflict_resolution, prefix_imported,
                    import_media, import_embeddings
                ))
                self._tasks[job_id] = task
                task.add_done_callback(lambda _t: self._tasks.pop(job_id, None))

            return True, f"Import job started: {job_id}", job_id
        else:
            if source_format_value == "openwebui_json":
                return await asyncio.to_thread(
                    self.import_openwebui_json,
                    str(resolved_path),
                    conflict_resolution,
                    prefix_imported,
                )
            if source_format_value == "openwebui_db":
                return await asyncio.to_thread(
                    self.import_openwebui_db,
                    str(resolved_path),
                    selected_user_id=str(selected_openwebui_user_id),
                    conflict_resolution=conflict_resolution,
                    prefix_imported=prefix_imported,
                )
            # Run synchronously (wrapped in executor for async compatibility)
            # Return (success, message, structured sync import result)
            return await asyncio.to_thread(
                self._import_chatbook_sync,
                str(resolved_path), content_selections,
                conflict_resolution, prefix_imported,
                import_media, import_embeddings
            )

    def _import_chatbook_sync(
        self,
        file_path: str,
        content_selections: dict[ContentType, list[str]] | None,
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        import_media: bool,
        import_embeddings: bool
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """
        Synchronously import a chatbook.
        """
        extract_dir: Path | None = None
        try:
            try:
                resolved_path = self._resolve_import_archive_path(file_path)
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"Chatbooks import rejected file path: {exc}")
                return False, "Invalid or potentially malicious archive file", None
            file_path = str(resolved_path)

            # Validate file first via centralized validator
            from .chatbook_validators import ChatbookValidator
            ok, err = ChatbookValidator.validate_zip_file(file_path)
            if not ok:
                # Surface specific validator detail
                detail = err or "Invalid or potentially malicious archive file"
                if isinstance(detail, str) and detail.lower().startswith("file does not exist"):
                    detail = "Invalid or potentially malicious archive file"
                if (
                    isinstance(detail, str)
                    and detail != "Invalid or potentially malicious archive file"
                    and "error" not in detail.lower()
                ):
                    detail = f"Error: {detail}"
                return False, detail, None

            # Extract chatbook to secure temp location
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extract_dir = self.temp_dir / f"import_{timestamp}_{uuid4().hex[:8]}"
            extract_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

            # Extract archive with size limits
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Check total uncompressed size (honor validator's configured limit)
                total_size = sum(zinfo.file_size for zinfo in zf.filelist)
                max_uncompressed = getattr(ChatbookValidator, "MAX_UNCOMPRESSED_SIZE", 500 * 1024 * 1024)
                if total_size > max_uncompressed:
                    max_mb = max_uncompressed / (1024 * 1024)
                    return False, f"Archive too large (> {max_mb:.0f}MB uncompressed)", None

                # Extract with path validation
                # Resolve extract_dir once (it exists at this point)
                extract_dir_resolved = str(extract_dir.resolve())

                for member in zf.namelist():
                    # Validate path using path-component aware check
                    if self._is_unsafe_archive_path(member):
                        return False, "Unsafe path in archive detected", None

                    # Additional check: ensure the normalized target stays within extract_dir
                    os.path.normpath(member)
                    # Use normpath + commonpath instead of realpath to avoid race conditions
                    # (realpath would try to resolve symlinks on paths that don't exist yet)
                    target_path = os.path.normpath(os.path.join(extract_dir_resolved, member))
                    try:
                        common = os.path.commonpath([extract_dir_resolved, target_path])
                        if common != extract_dir_resolved:
                            return False, "Path traversal attempt detected", None
                    except ValueError:
                        # commonpath raises ValueError if paths are on different drives (Windows)
                        return False, "Path traversal attempt detected", None

                    # Extract individual file safely
                    zf.extract(member, extract_dir)

            # Load manifest
            manifest_path = extract_dir / "manifest.json"
            if not manifest_path.exists():
                return False, "Invalid chatbook - manifest.json not found", None

            with open(manifest_path, encoding='utf-8') as f:
                manifest_data = json.load(f)

            manifest = ChatbookManifest.from_dict(manifest_data)

            # Check version compatibility (V1 and V1_LEGACY are both compatible)
            compatible_versions = {ChatbookVersion.V1, ChatbookVersion.V1_LEGACY}
            if manifest.version not in compatible_versions:
                logger.warning(f"Chatbook version {manifest.version.value} may not be fully compatible")

            # Set up content selections if not provided
            if content_selections is None:
                content_selections = {}
                for item in manifest.content_items:
                    if item.type not in content_selections:
                        content_selections[item.type] = []
                    content_selections[item.type].append(item.id)
            else:
                content_selections = dict(content_selections)

            # Import each content type
            import_status = ImportJob(
                job_id="temp",
                user_id=self.user_id,
                status=ImportStatus.IN_PROGRESS,
                chatbook_path=file_path
            )

            import_status.total_items = sum(len(ids) for ids in content_selections.values())

            supported_types = {
                ContentType.CHARACTER,
                ContentType.WORLD_BOOK,
                ContentType.DICTIONARY,
                ContentType.CONVERSATION,
                ContentType.NOTE,
            }
            unsupported_types = [ct for ct in content_selections if ct not in supported_types]
            for ct in unsupported_types:
                ids = content_selections.get(ct) or []
                if ids:
                    import_status.processed_items += len(ids)
                    import_status.skipped_items += len(ids)
                    label = ct.value if hasattr(ct, "value") else str(ct)
                    import_status.warnings.append(
                        f"Skipped unsupported content type '{label}' ({len(ids)} items)"
                    )
                content_selections.pop(ct, None)

            character_id_map: dict[str, int] = {}
            imported_items: dict[str, int] = {}

            def _run_import_for_type(
                content_type: ContentType,
                import_fn: Callable[..., Any],
                *args: Any,
                **kwargs: Any,
            ) -> None:
                before_successful = import_status.successful_items
                import_fn(*args, **kwargs)
                imported_count = import_status.successful_items - before_successful
                if imported_count > 0:
                    imported_items[content_type.value] = imported_count

            # Import characters first (they may be dependencies)
            if ContentType.CHARACTER in content_selections:
                _run_import_for_type(
                    ContentType.CHARACTER,
                    self._import_characters,
                    extract_dir, manifest,
                    content_selections[ContentType.CHARACTER],
                    conflict_resolution, prefix_imported,
                    import_status,
                    character_id_map=character_id_map
                )

            # Import world books
            if ContentType.WORLD_BOOK in content_selections:
                _run_import_for_type(
                    ContentType.WORLD_BOOK,
                    self._import_world_books,
                    extract_dir, manifest,
                    content_selections[ContentType.WORLD_BOOK],
                    conflict_resolution, prefix_imported,
                    import_status
                )

            # Import dictionaries
            if ContentType.DICTIONARY in content_selections:
                _run_import_for_type(
                    ContentType.DICTIONARY,
                    self._import_dictionaries,
                    extract_dir, manifest,
                    content_selections[ContentType.DICTIONARY],
                    conflict_resolution, prefix_imported,
                    import_status
                )

            # Import conversations
            if ContentType.CONVERSATION in content_selections:
                _run_import_for_type(
                    ContentType.CONVERSATION,
                    self._import_conversations,
                    extract_dir, manifest,
                    content_selections[ContentType.CONVERSATION],
                    conflict_resolution, prefix_imported,
                    import_status,
                    character_id_map=character_id_map
                )

            # Import notes
            if ContentType.NOTE in content_selections:
                _run_import_for_type(
                    ContentType.NOTE,
                    self._import_notes,
                    extract_dir, manifest,
                    content_selections[ContentType.NOTE],
                    conflict_resolution, prefix_imported,
                    import_status
                )

            # Note: We do NOT delete the original import file - the caller owns it

            # Build result message
            logger.debug(f"Import status: total={import_status.total_items}, successful={import_status.successful_items}, skipped={import_status.skipped_items}, failed={import_status.failed_items}")
            sync_result = {
                "imported_items": imported_items,
                "warnings": list(import_status.warnings or []),
            }

            if import_status.successful_items > 0:
                message = f"Successfully imported {import_status.successful_items}/{import_status.total_items} items"
                if import_status.skipped_items > 0:
                    message += f" ({import_status.skipped_items} skipped)"
                return True, message, sync_result
            elif import_status.total_items == 0:
                # No items to import is not an error
                return True, "Import completed: No items to import", sync_result
            elif import_status.skipped_items > 0:
                # All items were skipped (e.g., due to conflicts)
                return True, f"Import completed: All {import_status.skipped_items} items were skipped", sync_result
            else:
                logger.debug("Import failed: No items were successfully imported or skipped")
                return False, "No items were imported", sync_result

        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error importing chatbook: {e}")
            return False, f"Error importing chatbook: {str(e)}", None
        finally:
            if extract_dir and extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)

    async def _import_chatbook_async(
        self,
        job_id: str,
        file_path: str,
        content_selections: dict[ContentType, list[str]] | None,
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        import_media: bool,
        import_embeddings: bool
    ):
        """
        Asynchronously import a chatbook.
        """
        # Get job from database
        job = self._get_import_job(job_id)
        if not job:
            return

        try:
            # Update job status
            job.status = ImportStatus.IN_PROGRESS
            job.started_at = datetime.now(timezone.utc)
            self._save_import_job(job)
            if getattr(self, "_jobs_backend", "core") == "prompt_studio" and getattr(self, "_ps_job_adapter", None) is not None:
                try:
                    self._ps_job_adapter.update_status(int(job.job_id), "in_progress")
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"PS adapter update_status failed for import job {job.job_id}: {e}")

            # Import chatbook synchronously using thread pool
            success, message, _ = await asyncio.to_thread(
                self._import_chatbook_sync,
                file_path, content_selections,
                conflict_resolution, prefix_imported,
                import_media, import_embeddings
            )

            if success:
                latest = self._get_import_job(job.job_id)
                if latest and latest.status == ImportStatus.CANCELLED:
                    if getattr(self, "_jobs_backend", "core") == "prompt_studio" and getattr(self, "_ps_job_adapter", None) is not None:
                        try:
                            self._ps_job_adapter.update_status(int(job.job_id), "cancelled")
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                            logger.debug(f"PS adapter update_status (cancelled) failed for import job {job.job_id}: {e}")
                    return
                job.status = ImportStatus.COMPLETED
            else:
                job.status = ImportStatus.FAILED
                job.error_message = message

            job.completed_at = datetime.now(timezone.utc)
            self._save_import_job(job)
            if getattr(self, "_jobs_backend", "core") == "prompt_studio" and getattr(self, "_ps_job_adapter", None) is not None:
                try:
                    if job.status == ImportStatus.COMPLETED:
                        self._ps_job_adapter.update_status(int(job.job_id), "completed")
                    elif job.status == ImportStatus.FAILED:
                        self._ps_job_adapter.update_status(int(job.job_id), "failed", error_message=job.error_message)
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"PS adapter update_status (completion) failed for import job {job.job_id}: {e}")

        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            job.status = ImportStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error_message = str(e)
            self._save_import_job(job)
            if getattr(self, "_jobs_backend", "core") == "prompt_studio" and getattr(self, "_ps_job_adapter", None) is not None:
                try:
                    self._ps_job_adapter.update_status(int(job.job_id), "failed", error_message=str(e))
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as ps_err:
                    logger.debug(f"PS adapter update_status (failed) failed for import job {job.job_id}: {ps_err}")
        finally:
            # Ensure original import archive is removed for async mode
            try:
                fp = self._resolve_import_archive_path(file_path)
                if fp.exists() and fp.is_file():
                    fp.unlink()
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as _e:
                logger.warning(f"Could not remove import archive (async) {file_path}: {_e}")

    def _openwebui_duplicate_exists(self, external_ref: str) -> bool:
        """Return whether an OpenWebUI conversation was already imported for this client."""
        try:
            return bool(
                self.db.get_conversation_by_source_ref(
                    "openwebui",
                    external_ref,
                    client_id=getattr(self.db, "client_id", None),
                )
            )
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"OpenWebUI duplicate lookup failed for source ref {external_ref}: {exc}")
            return False

    def preview_openwebui_json(self, file_path: str) -> tuple[dict[str, Any] | None, str | None]:
        """Preview an OpenWebUI chat export JSON file without writing to the database."""
        try:
            resolved_path = self._resolve_import_upload_path(file_path)
            preview = preview_openwebui_export(
                resolved_path,
                duplicate_lookup=self._openwebui_duplicate_exists,
            )
            return preview.to_dict(), None
        except ValueError as exc:
            return None, str(exc)
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"OpenWebUI preview rejected file path: {exc}")
            return None, "Invalid or potentially malicious import file"

    def preview_openwebui_db(self, file_path: str) -> tuple[dict[str, Any] | None, str | None]:
        """Preview an OpenWebUI SQLite database file without writing to the database."""
        try:
            resolved_path = self._resolve_import_upload_path(file_path)
            preview = build_openwebui_db_preview(
                resolved_path,
                duplicate_lookup=self._openwebui_duplicate_exists,
            )
            return preview.to_dict(), None
        except ValueError as exc:
            return None, str(exc)
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"OpenWebUI DB preview rejected file path: {exc}")
            return None, "Invalid or potentially malicious import file"

    @staticmethod
    def _openwebui_timestamp_to_iso(value: Any) -> tuple[str, str | None]:
        """Convert OpenWebUI timestamps to UTC ISO strings."""
        if value is None or value == "":
            return datetime.now(timezone.utc).isoformat(), "OpenWebUI message missing timestamp; using import time."
        if isinstance(value, (int, float)):
            try:
                seconds = value / 1000 if value > 1_000_000_000_000 else value
                return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(), None
            except (OSError, OverflowError, ValueError):
                return datetime.now(timezone.utc).isoformat(), "OpenWebUI message has invalid timestamp; using import time."
        if isinstance(value, str):
            parsed = ChatbookService._parse_timestamp(value)
            if parsed is not None:
                return parsed.replace(tzinfo=timezone.utc).isoformat(), None
        return datetime.now(timezone.utc).isoformat(), "OpenWebUI message has unsupported timestamp; using import time."

    @staticmethod
    def _ordered_openwebui_messages(
        chat: OpenWebUIConversationPlan,
    ) -> tuple[list[OpenWebUIMessagePlan], list[str]]:
        """Order messages so parents are inserted before children."""
        remaining = {message.source_id: message for message in chat.messages}
        ordered: list[OpenWebUIMessagePlan] = []
        inserted: set[str] = set()
        warnings: list[str] = []

        while remaining:
            progressed = False
            for source_id, message in list(remaining.items()):
                parent_id = message.parent_source_id
                if not parent_id or (parent_id not in remaining and parent_id in inserted):
                    ordered.append(message)
                    inserted.add(source_id)
                    del remaining[source_id]
                    progressed = True
                    continue
                if parent_id and parent_id not in remaining and parent_id not in inserted:
                    warnings.append(
                        f"OpenWebUI message {source_id} references missing parent {parent_id}; importing as root."
                    )
                    ordered.append(message)
                    inserted.add(source_id)
                    del remaining[source_id]
                    progressed = True
                    continue
            if not progressed:
                skipped_ids = ", ".join(sorted(remaining))
                warnings.append(f"OpenWebUI message cycle or unresolved parent dependency skipped: {skipped_ids}")
                break

        return ordered, warnings

    @staticmethod
    def _openwebui_message_metadata(message: OpenWebUIMessagePlan) -> dict[str, Any]:
        """Build namespaced per-message OpenWebUI import metadata."""
        return {
            "openwebui_import": {
                "source_message_id": message.source_id,
                "source_parent_id": message.parent_source_id,
                "source_children_ids": list(message.children_source_ids),
                "role": message.role,
                "model": message.model,
                "attachment_refs": list(message.attachment_refs),
                "metadata": dict(message.metadata),
            }
        }

    def _store_openwebui_conversation_settings(
        self,
        conversation_id: str,
        chat: OpenWebUIConversationPlan,
        external_ref: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Merge OpenWebUI import metadata into conversation settings."""
        current = self.db.get_conversation_settings(conversation_id) or {}
        settings = current.get("settings") if isinstance(current, dict) else {}
        if not isinstance(settings, dict):
            settings = {}
        metadata = dict(chat.source_metadata)
        if extra_metadata:
            metadata.update(extra_metadata)
        merged = dict(settings)
        merged["openwebui_import"] = {
            "source": "openwebui",
            "external_ref": external_ref,
            "history_current_id": chat.history_current_id,
            "branched": chat.is_branched,
            "metadata": metadata,
        }
        return bool(self.db.upsert_conversation_settings(conversation_id, merged))

    def _rollback_openwebui_conversation(
        self,
        conversation_id: str,
        external_ref: str,
        warnings: list[str],
    ) -> None:
        """Remove a partially-created OpenWebUI conversation after a chat-level failure."""
        try:
            if not hasattr(self.db, "hard_delete_conversation"):
                warnings.append(
                    f"OpenWebUI chat {external_ref} failed after conversation creation; rollback is unavailable."
                )
                return
            if not self.db.hard_delete_conversation(conversation_id):
                warnings.append(
                    f"OpenWebUI chat {external_ref} failed after conversation creation; partial conversation cleanup did not remove a row."
                )
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
            warnings.append(
                f"OpenWebUI chat {external_ref} failed after conversation creation; rollback failed: {exc}"
            )

    def import_openwebui_json(
        self,
        file_path: str,
        conflict_resolution: ConflictResolution | str | None = None,
        prefix_imported: bool = False,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Synchronously import an OpenWebUI chat export JSON file."""
        if isinstance(conflict_resolution, str):
            try:
                conflict_resolution = ConflictResolution(conflict_resolution)
            except (ValueError, KeyError):
                conflict_resolution = ConflictResolution.SKIP
        elif conflict_resolution is None:
            conflict_resolution = ConflictResolution.SKIP

        if conflict_resolution not in {ConflictResolution.SKIP, ConflictResolution.RENAME}:
            return False, "OpenWebUI imports support only skip or rename conflict handling.", None

        fallback_character_id = self._get_fallback_character_id()
        if fallback_character_id is None:
            return False, "OpenWebUI import requires at least one fallback character.", None

        try:
            resolved_path = self._resolve_import_upload_path(file_path)
            parsed = load_openwebui_export(resolved_path)
        except ValueError as exc:
            return False, str(exc), None
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"OpenWebUI import rejected file path: {exc}")
            return False, "Invalid or potentially malicious import file", None

        result = {
            "imported_chats": 0,
            "skipped_chats": 0,
            "failed_chats": 0,
            "imported_messages": 0,
            "skipped_messages": 0,
            "duplicate_chats": 0,
            "warnings": list(parsed.warnings),
        }

        for chat in parsed.chats:
            import_external_ref = chat.external_ref
            conversation_id: str | None = None
            chat_imported_messages = 0
            chat_skipped_messages = 0
            chat_warnings: list[str] = []
            try:
                existing = self.db.get_conversation_by_source_ref(
                    "openwebui",
                    import_external_ref,
                    client_id=getattr(self.db, "client_id", None),
                )
                if existing:
                    result["duplicate_chats"] += 1
                    if conflict_resolution == ConflictResolution.SKIP:
                        result["skipped_chats"] += 1
                        continue
                    import_external_ref = f"{chat.external_ref}#copy:{uuid4().hex[:8]}"

                conversation_title = f"[Imported] {chat.title}" if prefix_imported else chat.title
                if existing and conflict_resolution == ConflictResolution.RENAME:
                    conversation_title = self._generate_openwebui_copy_title(conversation_title)

                ordered_messages, order_warnings = self._ordered_openwebui_messages(chat)
                chat_warnings.extend(order_warnings)
                importable_messages = [
                    message
                    for message in ordered_messages
                    if (message.role or "").lower() in {"user", "assistant"} and message.content.strip()
                ]
                skipped_for_role = len(ordered_messages) - len(importable_messages)
                if skipped_for_role:
                    chat_skipped_messages += skipped_for_role
                    chat_warnings.append(
                        f"OpenWebUI chat {chat.external_ref} skipped {skipped_for_role} unsupported or empty messages."
                    )
                if not importable_messages:
                    result["failed_chats"] += 1
                    result["skipped_messages"] += chat_skipped_messages
                    result["warnings"].extend(chat_warnings)
                    result["warnings"].append(f"OpenWebUI chat {chat.external_ref} has no importable messages.")
                    continue

                conversation_id = self.db.add_conversation(
                    {
                        "title": conversation_title,
                        "character_id": fallback_character_id,
                        "source": "openwebui",
                        "external_ref": import_external_ref,
                        "client_id": getattr(self.db, "client_id", None),
                    }
                )
                if not conversation_id:
                    result["failed_chats"] += 1
                    result["skipped_messages"] += chat_skipped_messages
                    result["warnings"].extend(chat_warnings)
                    result["warnings"].append(f"OpenWebUI chat {chat.external_ref} could not create a conversation.")
                    continue

                if not self._store_openwebui_conversation_settings(conversation_id, chat, import_external_ref):
                    raise DatabaseError(
                        f"OpenWebUI chat {chat.external_ref} conversation metadata was not stored."
                    )

                message_id_map: dict[str, str] = {}
                namespace = f"tldw-openwebui:{import_external_ref}"
                for message in importable_messages:
                    parent_message_id = None
                    if message.parent_source_id:
                        parent_message_id = message_id_map.get(message.parent_source_id)
                        if parent_message_id is None:
                            chat_warnings.append(
                                f"OpenWebUI message {message.source_id} imported as root because its parent was not imported."
                            )
                    timestamp, timestamp_warning = self._openwebui_timestamp_to_iso(message.timestamp)
                    if timestamp_warning:
                        chat_warnings.append(f"{timestamp_warning} source_message_id={message.source_id}")
                    message_id = str(uuid5(NAMESPACE_URL, f"{namespace}:{message.source_id}"))
                    inserted_message_id = self.db.add_message(
                        {
                            "id": message_id,
                            "conversation_id": conversation_id,
                            "parent_message_id": parent_message_id,
                            "sender": (message.role or "user").lower(),
                            "content": message.content,
                            "timestamp": timestamp,
                            "client_id": getattr(self.db, "client_id", None),
                        }
                    )
                    if not inserted_message_id:
                        raise DatabaseError(
                            f"OpenWebUI message {message.source_id} could not be stored."
                        )
                    stored_message_id = str(inserted_message_id)
                    message_id_map[message.source_id] = stored_message_id
                    chat_imported_messages += 1
                    if not self.db.set_message_metadata_extra(
                        stored_message_id,
                        self._openwebui_message_metadata(message),
                        merge=True,
                    ):
                        raise DatabaseError(
                            f"OpenWebUI message {message.source_id} metadata was not stored."
                        )

                if chat_imported_messages == 0:
                    self._rollback_openwebui_conversation(conversation_id, chat.external_ref, chat_warnings)
                    result["failed_chats"] += 1
                    result["skipped_messages"] += chat_skipped_messages
                    result["warnings"].extend(chat_warnings)
                    result["warnings"].append(
                        f"OpenWebUI chat {chat.external_ref} imported no messages and was rolled back."
                    )
                    continue

                result["imported_messages"] += chat_imported_messages
                result["skipped_messages"] += chat_skipped_messages
                result["warnings"].extend(chat_warnings)
                result["imported_chats"] += 1
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                if conversation_id:
                    self._rollback_openwebui_conversation(conversation_id, chat.external_ref, chat_warnings)
                result["failed_chats"] += 1
                result["skipped_messages"] += chat_skipped_messages
                result["warnings"].extend(chat_warnings)
                result["warnings"].append(f"OpenWebUI chat {chat.external_ref} failed to import: {exc}")

        if parsed.malformed_chat_count:
            result["failed_chats"] += parsed.malformed_chat_count

        return True, "OpenWebUI import completed", result

    def import_openwebui_db(
        self,
        file_path: str,
        *,
        selected_user_id: str,
        conflict_resolution: ConflictResolution | str | None = None,
        prefix_imported: bool = False,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Synchronously import one selected user from an OpenWebUI SQLite database."""
        if isinstance(conflict_resolution, str):
            try:
                conflict_resolution = ConflictResolution(conflict_resolution)
            except (ValueError, KeyError):
                conflict_resolution = ConflictResolution.SKIP
        elif conflict_resolution is None:
            conflict_resolution = ConflictResolution.SKIP

        if conflict_resolution not in {ConflictResolution.SKIP, ConflictResolution.RENAME}:
            return False, "OpenWebUI imports support only skip or rename conflict handling.", None
        if not (selected_user_id or "").strip():
            return False, "selected_user_id is required for OpenWebUI database imports.", None

        fallback_character_id = self._get_fallback_character_id()
        if fallback_character_id is None:
            return False, "OpenWebUI import requires at least one fallback character.", None

        try:
            resolved_path = self._resolve_import_upload_path(file_path)
            extracted = extract_openwebui_db_user(
                resolved_path,
                selected_user_id=selected_user_id,
            )
        except ValueError as exc:
            return False, str(exc), None
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"OpenWebUI DB import rejected file path: {exc}")
            return False, "Invalid or potentially malicious import file", None

        result = {
            "selected_user_id": extracted.selected_user_id,
            "selected_user_label": extracted.selected_user_label,
            "imported_chats": 0,
            "skipped_chats": 0,
            "failed_chats": 0,
            "imported_messages": 0,
            "skipped_messages": 0,
            "duplicate_chats": 0,
            "mirrored_folders": 0,
            "folder_links": 0,
            "warnings": list(extracted.warnings),
        }
        mirrored_folder_ids: set[int] = set()
        namespace_segments = build_openwebui_namespace_segments(
            extracted.selected_user_label,
            extracted.selected_user_id,
        )

        for chat in extracted.chats:
            import_external_ref = chat.external_ref
            conversation_id: str | None = None
            chat_imported_messages = 0
            chat_skipped_messages = 0
            chat_warnings: list[str] = []
            try:
                existing = self.db.get_conversation_by_source_ref(
                    "openwebui",
                    import_external_ref,
                    client_id=getattr(self.db, "client_id", None),
                )
                if existing:
                    result["duplicate_chats"] += 1
                    if conflict_resolution == ConflictResolution.SKIP:
                        result["skipped_chats"] += 1
                        continue
                    import_external_ref = f"{chat.external_ref}#copy:{uuid4().hex[:8]}"

                conversation_title = f"[Imported] {chat.title}" if prefix_imported else chat.title
                if existing and conflict_resolution == ConflictResolution.RENAME:
                    conversation_title = self._generate_openwebui_copy_title(conversation_title)

                ordered_messages, order_warnings = self._ordered_openwebui_messages(chat)
                chat_warnings.extend(order_warnings)
                importable_messages = [
                    message
                    for message in ordered_messages
                    if (message.role or "").lower() in {"user", "assistant"} and message.content.strip()
                ]
                skipped_for_role = len(ordered_messages) - len(importable_messages)
                if skipped_for_role:
                    chat_skipped_messages += skipped_for_role
                    chat_warnings.append(
                        f"OpenWebUI chat {chat.external_ref} skipped {skipped_for_role} unsupported or empty messages."
                    )
                if not importable_messages:
                    result["failed_chats"] += 1
                    result["skipped_messages"] += chat_skipped_messages
                    result["warnings"].extend(chat_warnings)
                    result["warnings"].append(f"OpenWebUI chat {chat.external_ref} has no importable messages.")
                    continue

                conversation_id = self.db.add_conversation(
                    {
                        "title": conversation_title,
                        "character_id": fallback_character_id,
                        "source": "openwebui",
                        "external_ref": import_external_ref,
                        "client_id": getattr(self.db, "client_id", None),
                    }
                )
                if not conversation_id:
                    result["failed_chats"] += 1
                    result["skipped_messages"] += chat_skipped_messages
                    result["warnings"].extend(chat_warnings)
                    result["warnings"].append(f"OpenWebUI chat {chat.external_ref} could not create a conversation.")
                    continue

                folder_plan = extracted.folder_plans_by_external_ref.get(chat.external_ref)
                folder_metadata: dict[str, Any] = {}
                if folder_plan is not None:
                    folder_metadata = {
                        "folder_path": list(folder_plan.source_path),
                        "folder_source_parent_id": folder_plan.source_parent_id,
                        "folder_source_meta": dict(folder_plan.source_meta),
                    }

                if not self._store_openwebui_conversation_settings(
                    conversation_id,
                    chat,
                    import_external_ref,
                    extra_metadata=folder_metadata,
                ):
                    raise DatabaseError(
                        f"OpenWebUI chat {chat.external_ref} conversation metadata was not stored."
                    )

                message_id_map: dict[str, str] = {}
                namespace = f"tldw-openwebui:{import_external_ref}"
                for message in importable_messages:
                    parent_message_id = None
                    if message.parent_source_id:
                        parent_message_id = message_id_map.get(message.parent_source_id)
                        if parent_message_id is None:
                            chat_warnings.append(
                                f"OpenWebUI message {message.source_id} imported as root because its parent was not imported."
                            )
                    timestamp, timestamp_warning = self._openwebui_timestamp_to_iso(message.timestamp)
                    if timestamp_warning:
                        chat_warnings.append(f"{timestamp_warning} source_message_id={message.source_id}")
                    message_id = str(uuid5(NAMESPACE_URL, f"{namespace}:{message.source_id}"))
                    inserted_message_id = self.db.add_message(
                        {
                            "id": message_id,
                            "conversation_id": conversation_id,
                            "parent_message_id": parent_message_id,
                            "sender": (message.role or "user").lower(),
                            "content": message.content,
                            "timestamp": timestamp,
                            "client_id": getattr(self.db, "client_id", None),
                        }
                    )
                    if not inserted_message_id:
                        raise DatabaseError(
                            f"OpenWebUI message {message.source_id} could not be stored."
                        )
                    stored_message_id = str(inserted_message_id)
                    message_id_map[message.source_id] = stored_message_id
                    chat_imported_messages += 1
                    if not self.db.set_message_metadata_extra(
                        stored_message_id,
                        self._openwebui_message_metadata(message),
                        merge=True,
                    ):
                        raise DatabaseError(
                            f"OpenWebUI message {message.source_id} metadata was not stored."
                        )

                if chat_imported_messages == 0:
                    self._rollback_openwebui_conversation(conversation_id, chat.external_ref, chat_warnings)
                    result["failed_chats"] += 1
                    result["skipped_messages"] += chat_skipped_messages
                    result["warnings"].extend(chat_warnings)
                    result["warnings"].append(
                        f"OpenWebUI chat {chat.external_ref} imported no messages and was rolled back."
                    )
                    continue

                source_path = list(folder_plan.source_path) if folder_plan is not None else ["Unfiled"]
                source_folder_id = folder_plan.source_folder_id if folder_plan is not None else None
                source_meta = dict(folder_plan.source_meta) if folder_plan is not None else {}
                try:
                    mirror_result = mirror_openwebui_folder_for_conversation(
                        self.db,
                        conversation_id=conversation_id,
                        namespace_segments=namespace_segments,
                        source_path_segments=source_path,
                        source_folder_id=source_folder_id,
                        metadata={
                            "source_user_id": extracted.selected_user_id,
                            "external_ref": chat.external_ref,
                            "import_external_ref": import_external_ref,
                            "source_meta": source_meta,
                        },
                    )
                    if mirror_result.final_collection_id is not None:
                        mirrored_folder_ids.add(mirror_result.final_collection_id)
                        result["mirrored_folders"] = len(mirrored_folder_ids)
                    if mirror_result.conversation_keyword_linked:
                        result["folder_links"] += 1
                    chat_warnings.extend(mirror_result.warnings)
                except _OPENWEBUI_FOLDER_MIRROR_EXCEPTIONS as exc:
                    chat_warnings.append(
                        f"OpenWebUI chat {chat.external_ref} folder mirroring failed: {exc}"
                    )

                result["imported_messages"] += chat_imported_messages
                result["skipped_messages"] += chat_skipped_messages
                result["warnings"].extend(chat_warnings)
                result["imported_chats"] += 1
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                if conversation_id:
                    self._rollback_openwebui_conversation(conversation_id, chat.external_ref, chat_warnings)
                result["failed_chats"] += 1
                result["skipped_messages"] += chat_skipped_messages
                result["warnings"].extend(chat_warnings)
                result["warnings"].append(f"OpenWebUI chat {chat.external_ref} failed to import: {exc}")

        return True, "OpenWebUI database import completed", result

    def _openwebui_conversation_title_exists(self, title: str) -> bool:
        """Return whether an exact conversation title already exists for the current client."""
        try:
            if hasattr(self.db, "conversation_title_exists"):
                return bool(
                    self.db.conversation_title_exists(
                        title,
                        client_id=getattr(self.db, "client_id", None),
                    )
                )
            conversation = self._get_conversation_by_name(title)
            return bool(conversation and conversation.get("title") == title and not conversation.get("deleted"))
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"OpenWebUI import could not check existing conversation title: {exc}")
            return False

    def _generate_openwebui_copy_title(self, base_title: str) -> str:
        """Generate an OpenWebUI duplicate-copy title without FTS title search."""
        for counter in range(1, 1001):
            candidate = f"{base_title} ({counter})"
            if not self._openwebui_conversation_title_exists(candidate):
                return candidate
        raise ValueError("Could not generate unique OpenWebUI conversation title after 1000 attempts")

    def preview_chatbook(self, file_path: str) -> tuple[ChatbookManifest | None, str | None]:
        """
        Preview a chatbook without importing it.

        Args:
            file_path: Path to chatbook file

        Returns:
            Tuple of (manifest, error_message)
        """
        extract_dir: Path | None = None
        try:
            try:
                resolved_path = self._resolve_import_archive_path(file_path)
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"Chatbooks preview rejected file path: {exc}")
                return None, "Invalid or potentially malicious archive file"
            file_path = str(resolved_path)

            # Defense-in-depth: validate the archive before extraction.
            # Any unexpected validator fault should surface as a server error.
            from .chatbook_validators import ChatbookValidator
            ok, err = ChatbookValidator.validate_zip_file(file_path)
            if not ok:
                return None, err or "Invalid archive"
            # Extract to temporary directory with UUID to prevent collisions
            extract_dir = self.temp_dir / f"preview_{uuid4().hex}"

            # Extract archive with path validation
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Ensure extract_dir exists for path validation
                extract_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                extract_dir_resolved = str(extract_dir.resolve())

                # Validate all paths before extraction to prevent path traversal
                for member in zf.namelist():
                    # Validate path using path-component aware check
                    if self._is_unsafe_archive_path(member):
                        return None, "Unsafe path in archive detected"

                    # Additional check: ensure the normalized target stays within extract_dir
                    os.path.normpath(member)
                    # Use normpath + commonpath instead of realpath to avoid race conditions
                    target_path = os.path.normpath(os.path.join(extract_dir_resolved, member))
                    try:
                        common = os.path.commonpath([extract_dir_resolved, target_path])
                        if common != extract_dir_resolved:
                            return None, "Path traversal attempt detected"
                    except ValueError:
                        # commonpath raises ValueError if paths are on different drives (Windows)
                        return None, "Path traversal attempt detected"

                # Safe to extract after validation
                zf.extractall(extract_dir)

            # Load manifest
            manifest_path = extract_dir / "manifest.json"
            if not manifest_path.exists():
                return None, "Invalid chatbook: manifest.json not found"

            try:
                with open(manifest_path, encoding='utf-8') as f:
                    manifest_data = json.load(f)
                manifest = ChatbookManifest.from_dict(manifest_data)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, ValidationError):
                return None, "Invalid chatbook manifest"

            return manifest, None

        except zipfile.BadZipFile:
            return None, "Invalid archive"
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error previewing chatbook: {e}")
            raise
        finally:
            if extract_dir and extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)

    def _build_download_url(self, job_id: str, expires_at: datetime | None) -> str:
        """Build a (possibly signed) download URL for a job."""
        base = f"/api/v1/chatbooks/download/{job_id}"
        use_signed = str(os.getenv("CHATBOOKS_SIGNED_URLS", "false")).lower() in {"1","true","yes"}
        secret = os.getenv("CHATBOOKS_SIGNING_SECRET", "")
        if use_signed and secret and expires_at:
            import hashlib
            import hmac
            exp = int(expires_at.timestamp())
            msg = f"{job_id}:{exp}".encode()
            sig = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
            return f"{base}?exp={exp}&token={sig}"
        return base

    def get_export_job(self, job_id: str) -> ExportJob | None:
        """Get export job status."""
        job = self._get_export_job(job_id)
        # If using PS backend, reflect PS status for live view
        if job and getattr(self, "_jobs_backend", "core") == "prompt_studio" and getattr(self, "_ps_job_adapter", None) is not None:
            try:
                ps_id = int(job_id)
                ps_job = self._ps_job_adapter.get(ps_id)
                if ps_job and isinstance(ps_job, dict):
                    ps_status = str(ps_job.get("status", "")).lower()
                    status_map = {
                        "queued": ExportStatus.PENDING,
                        "processing": ExportStatus.IN_PROGRESS,
                        "completed": ExportStatus.COMPLETED,
                        "failed": ExportStatus.FAILED,
                        "cancelled": ExportStatus.CANCELLED,
                    }
                    mapped = status_map.get(ps_status)
                    if mapped and job.status not in {ExportStatus.COMPLETED, ExportStatus.FAILED}:
                        job.status = mapped
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                pass
        if job and getattr(self, "_jobs_backend", "core") == "core" and getattr(self, "_jobs_adapter", None) is not None:
            with contextlib.suppress(_CHATBOOK_NONCRITICAL_EXCEPTIONS):
                self._jobs_adapter.apply_export_status(job)
        return job

    def get_import_job(self, job_id: str) -> ImportJob | None:
        """Get import job status."""
        job = self._get_import_job(job_id)
        if job and getattr(self, "_jobs_backend", "core") == "prompt_studio" and getattr(self, "_ps_job_adapter", None) is not None:
            try:
                ps_id = int(job_id)
                ps_job = self._ps_job_adapter.get(ps_id)
                if ps_job and isinstance(ps_job, dict):
                    ps_status = str(ps_job.get("status", "")).lower()
                    status_map = {
                        "queued": ImportStatus.PENDING,
                        "processing": ImportStatus.IN_PROGRESS,
                        "completed": ImportStatus.COMPLETED,
                        "failed": ImportStatus.FAILED,
                        "cancelled": ImportStatus.CANCELLED,
                    }
                    mapped = status_map.get(ps_status)
                    if mapped and job.status not in {ImportStatus.COMPLETED, ImportStatus.FAILED}:
                        job.status = mapped
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                pass
        if job and getattr(self, "_jobs_backend", "core") == "core" and getattr(self, "_jobs_adapter", None) is not None:
            with contextlib.suppress(_CHATBOOK_NONCRITICAL_EXCEPTIONS):
                self._jobs_adapter.apply_import_status(job)
        return job

    def list_export_jobs(self, status: str | None = None, limit: int = 100, offset: int = 0) -> list[ExportJob]:
        """List all export jobs for this user.

        Args:
            status: Optional status filter (pending, in_progress, completed, failed, cancelled, expired)
            limit: Maximum number of jobs to return
            offset: Offset for pagination
        """
        # Sanity check: ensure user_id is set to prevent listing all jobs
        if not self.user_id:
            logger.warning("list_export_jobs called with empty user_id")
            return []

        try:
            # Normalize pagination inputs
            try:
                limit = int(limit)
            except (TypeError, ValueError):
                limit = 100
            try:
                offset = int(offset)
            except (TypeError, ValueError):
                offset = 0
            if limit <= 0:
                limit = 100
            if offset < 0:
                offset = 0

            # Build query with optional status filter
            query = "SELECT * FROM export_jobs WHERE user_id = ?"
            params: list = [self.user_id]

            if status:
                # Validate status to prevent SQL injection
                valid_statuses = {'pending', 'in_progress', 'completed', 'failed', 'cancelled', 'expired'}
                if status.lower() in valid_statuses:
                    query += " AND status = ?"
                    params.append(status.lower())

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor = self.db.execute_query(query, tuple(params))

            # Fetch results from cursor
            results = self._fetch_results(cursor)

            if not results:
                return []

            jobs: list[ExportJob] = []
            for row in results:
                # Handle both dict and tuple formats (for test compatibility)
                if isinstance(row, dict):
                    # Use class method for timestamp parsing
                    job = ExportJob(
                        job_id=row['job_id'],
                        user_id=row['user_id'],
                        status=ExportStatus(row['status']),
                        chatbook_name=row['chatbook_name'],
                        output_path=row['output_path'],
                        created_at=ChatbookService._parse_timestamp(row['created_at']),
                        started_at=ChatbookService._parse_timestamp(row['started_at']),
                        completed_at=ChatbookService._parse_timestamp(row['completed_at']),
                        error_message=row['error_message'],
                        progress_percentage=row['progress_percentage'] or 0,
                        total_items=row['total_items'] or 0,
                        processed_items=row['processed_items'] or 0,
                        file_size_bytes=row['file_size_bytes'],
                        download_url=row['download_url'],
                        expires_at=ChatbookService._parse_timestamp(row['expires_at']),
                        metadata={}  # Initialize empty metadata
                    )
                else:
                    # Handle tuple format from mocked tests
                    # (job_id, user_id, status, chatbook_name, output_path, created_at,
                    #  started_at, completed_at, error_message, progress_percentage,
                    #  total_items, processed_items, file_size_bytes, download_url, expires_at)
                    job = ExportJob(
                        job_id=row[0],
                        user_id=row[1],
                        status=ExportStatus(row[2]),
                        chatbook_name=row[3],
                        output_path=row[4],
                        created_at=ChatbookService._parse_timestamp(row[5]),
                        started_at=ChatbookService._parse_timestamp(row[6]),
                        completed_at=ChatbookService._parse_timestamp(row[7]),
                        error_message=row[8] if len(row) > 8 else None,
                        progress_percentage=row[9] if len(row) > 9 else 0,
                        total_items=row[10] if len(row) > 10 else 0,
                        processed_items=row[11] if len(row) > 11 else 0,
                        file_size_bytes=row[12] if len(row) > 12 else 0,
                        download_url=row[13] if len(row) > 13 else None,
                        expires_at=ChatbookService._parse_timestamp(row[14]) if len(row) > 14 else None,
                        metadata={}  # Initialize empty metadata
                    )
                # Reflect PS status if applicable
                if getattr(self, "_jobs_backend", "core") == "prompt_studio" and getattr(self, "_ps_job_adapter", None) is not None:
                    try:
                        ps_id = int(job.job_id)
                        ps_job = self._ps_job_adapter.get(ps_id)
                        if ps_job and isinstance(ps_job, dict):
                            ps_status = str(ps_job.get("status", "")).lower()
                            status_map = {
                                "queued": ExportStatus.PENDING,
                                "processing": ExportStatus.IN_PROGRESS,
                                "completed": ExportStatus.COMPLETED,
                                "failed": ExportStatus.FAILED,
                                "cancelled": ExportStatus.CANCELLED,
                            }
                            mapped = status_map.get(ps_status)
                            if mapped and job.status not in {ExportStatus.COMPLETED, ExportStatus.FAILED}:
                                job.status = mapped
                    except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                        pass
                jobs.append(job)

            if getattr(self, "_jobs_backend", "core") == "core" and getattr(self, "_jobs_adapter", None) is not None:
                try:
                    job_ids = [job.job_id for job in jobs]
                    job_map = self._jobs_adapter.map_jobs(job_ids=job_ids, job_type="export", limit=len(job_ids) or 1)
                    for job in jobs:
                        row = job_map.get(job.job_id)
                        if row:
                            self._jobs_adapter.apply_export_status(job, job_row=row)
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                    pass

            return jobs
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error listing export jobs: {e}")
            return []

    def list_import_jobs(self, status: str | None = None, limit: int = 100, offset: int = 0) -> list[ImportJob]:
        """List all import jobs for this user.

        Args:
            status: Optional status filter (pending, validating, in_progress, completed, failed, cancelled)
            limit: Maximum number of jobs to return
            offset: Offset for pagination
        """
        # Sanity check: ensure user_id is set to prevent listing all jobs
        if not self.user_id:
            logger.warning("list_import_jobs called with empty user_id")
            return []

        try:
            # Normalize pagination inputs
            try:
                limit = int(limit)
            except (TypeError, ValueError):
                limit = 100
            try:
                offset = int(offset)
            except (TypeError, ValueError):
                offset = 0
            if limit <= 0:
                limit = 100
            if offset < 0:
                offset = 0

            # Build query with optional status filter
            query = "SELECT * FROM import_jobs WHERE user_id = ?"
            params: list = [self.user_id]

            if status:
                # Validate status to prevent SQL injection
                valid_statuses = {'pending', 'validating', 'in_progress', 'completed', 'failed', 'cancelled'}
                if status.lower() in valid_statuses:
                    query += " AND status = ?"
                    params.append(status.lower())

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor = self.db.execute_query(query, tuple(params))

            # Fetch results from cursor
            results = self._fetch_results(cursor)

            if not results:
                return []

            jobs: list[ImportJob] = []
            for row in results:
                # Handle both dict and tuple formats (for test compatibility)
                if isinstance(row, dict):
                    # Use class method for timestamp parsing
                    job = ImportJob(
                        job_id=row['job_id'],
                        user_id=row['user_id'],
                        status=ImportStatus(row['status']),
                        chatbook_path=row['chatbook_path'],
                        created_at=ChatbookService._parse_timestamp(row['created_at']),
                        started_at=ChatbookService._parse_timestamp(row['started_at']),
                        completed_at=ChatbookService._parse_timestamp(row['completed_at']),
                        error_message=row['error_message'],
                        progress_percentage=row['progress_percentage'] or 0,
                        total_items=row['total_items'] or 0,
                        processed_items=row['processed_items'] or 0,
                        successful_items=row['successful_items'] or 0,
                        failed_items=row['failed_items'] or 0,
                        skipped_items=row['skipped_items'] or 0,
                        conflicts=json.loads(row['conflicts']) if row['conflicts'] else [],
                        warnings=json.loads(row['warnings']) if row['warnings'] else []
                    )
                else:
                    # Handle tuple format from mocked tests
                    # (job_id, user_id, status, chatbook_path, created_at, started_at,
                    #  completed_at, error_message, progress_percentage, total_items,
                    #  processed_items, successful_items, failed_items, skipped_items,
                    #  conflicts, warnings)
                    job = ImportJob(
                        job_id=row[0],
                        user_id=row[1],
                        status=ImportStatus(row[2]),
                        chatbook_path=row[3],
                        created_at=ChatbookService._parse_timestamp(row[4]),
                        started_at=ChatbookService._parse_timestamp(row[5]),
                        completed_at=ChatbookService._parse_timestamp(row[6]),
                        error_message=row[7] if len(row) > 7 else None,
                        progress_percentage=row[8] if len(row) > 8 else 0,
                        total_items=row[9] if len(row) > 9 else 0,
                        processed_items=row[10] if len(row) > 10 else 0,
                        successful_items=row[11] if len(row) > 11 else 0,
                        failed_items=row[12] if len(row) > 12 else 0,
                        skipped_items=row[13] if len(row) > 13 else 0,
                        conflicts=json.loads(row[14]) if len(row) > 14 and row[14] else [],
                        warnings=json.loads(row[15]) if len(row) > 15 and row[15] else []
                    )
                if getattr(self, "_jobs_backend", "core") == "prompt_studio" and getattr(self, "_ps_job_adapter", None) is not None:
                    try:
                        ps_id = int(job.job_id)
                        ps_job = self._ps_job_adapter.get(ps_id)
                        if ps_job and isinstance(ps_job, dict):
                            ps_status = str(ps_job.get("status", "")).lower()
                            status_map = {
                                "queued": ImportStatus.PENDING,
                                "processing": ImportStatus.IN_PROGRESS,
                                "completed": ImportStatus.COMPLETED,
                                "failed": ImportStatus.FAILED,
                                "cancelled": ImportStatus.CANCELLED,
                            }
                            mapped = status_map.get(ps_status)
                            if mapped and job.status not in {ImportStatus.COMPLETED, ImportStatus.FAILED}:
                                job.status = mapped
                    except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                        pass
                jobs.append(job)

            if getattr(self, "_jobs_backend", "core") == "core" and getattr(self, "_jobs_adapter", None) is not None:
                try:
                    job_ids = [job.job_id for job in jobs]
                    job_map = self._jobs_adapter.map_jobs(job_ids=job_ids, job_type="import", limit=len(job_ids) or 1)
                    for job in jobs:
                        row = job_map.get(job.job_id)
                        if row:
                            self._jobs_adapter.apply_import_status(job, job_row=row)
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                    pass

            return jobs
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error listing import jobs: {e}")
            return []

    def count_export_jobs(self, status: str | None = None) -> int:
        """Count export jobs for this user."""
        if not self.user_id:
            logger.warning("count_export_jobs called with empty user_id")
            return 0
        try:
            query = "SELECT COUNT(*) AS c FROM export_jobs WHERE user_id = ?"
            params: list = [self.user_id]
            if status:
                valid_statuses = {'pending', 'in_progress', 'completed', 'failed', 'cancelled', 'expired'}
                if status.lower() in valid_statuses:
                    query += " AND status = ?"
                    params.append(status.lower())
            cursor = self.db.execute_query(query, tuple(params))
            results = self._fetch_results(cursor)
            if not results:
                return 0
            row = results[0]
            if isinstance(row, dict):
                return int(row.get("c") or 0)
            return int(row[0]) if row else 0
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error counting export jobs: {e}")
            return 0

    def count_import_jobs(self, status: str | None = None) -> int:
        """Count import jobs for this user."""
        if not self.user_id:
            logger.warning("count_import_jobs called with empty user_id")
            return 0
        try:
            query = "SELECT COUNT(*) AS c FROM import_jobs WHERE user_id = ?"
            params: list = [self.user_id]
            if status:
                valid_statuses = {'pending', 'validating', 'in_progress', 'completed', 'failed', 'cancelled'}
                if status.lower() in valid_statuses:
                    query += " AND status = ?"
                    params.append(status.lower())
            cursor = self.db.execute_query(query, tuple(params))
            results = self._fetch_results(cursor)
            if not results:
                return 0
            row = results[0]
            if isinstance(row, dict):
                return int(row.get("c") or 0)
            return int(row[0]) if row else 0
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error counting import jobs: {e}")
            return 0

    def cleanup_expired_exports(self, batch_size: int = 100) -> int:
        """Clean up expired export files. Returns number of files deleted.

        Args:
            batch_size: Number of jobs to process per batch to prevent memory issues
        """
        try:
            # Get expired jobs in batches to prevent memory issues with large result sets
            # Use the same timestamp format as stored in the jobs table for lexicographic compare
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            now_str = now.strftime('%Y-%m-%d %H:%M:%S.%f')
            deleted_count = 0
            no_progress_batches = 0

            while True:
                cursor = self.db.execute_query(
                    "SELECT * FROM export_jobs WHERE user_id = ? AND expires_at < ? AND status = ? LIMIT ?",
                    (self.user_id, now_str, ExportStatus.COMPLETED.value, batch_size)
                )
                results = self._fetch_results(cursor)

                if not results:
                    break

                updated_status_count = 0
                for row in results:
                    # Support both dict and tuple rows
                    if isinstance(row, dict):
                        output_path = row.get('output_path')
                        job_id = row.get('job_id')
                    else:
                        # tuple field order: job_id, user_id, status, chatbook_name, output_path, ...
                        output_path = row[4] if len(row) > 4 else None
                        job_id = row[0]

                    if output_path:
                        try:
                            file_path = Path(output_path).resolve()
                            expected_base = Path(self.export_dir).resolve()
                            if os.path.commonpath([str(file_path), str(expected_base)]) != str(expected_base):
                                logger.warning(f"Refusing to delete export outside export dir: {file_path}")
                            elif file_path.exists() and file_path.is_file():
                                file_path.unlink()
                                deleted_count += 1
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                            logger.error(f"Error deleting expired export: {e}")

                    # Update job status
                    try:
                        self.db.execute_query(
                            "UPDATE export_jobs SET status = ? WHERE job_id = ?",
                            ('expired', job_id),
                            commit=True,
                        )
                        updated_status_count += 1
                    except _CHATBOOK_NONCRITICAL_EXCEPTIONS as _e:
                        logger.warning(f"Failed to mark job {job_id} expired: {_e}")

                if updated_status_count == 0:
                    no_progress_batches += 1
                    logger.warning(
                        f"cleanup_expired_exports made no progress for user={self.user_id} "
                        f"(batch_size={len(results)}, attempt={no_progress_batches})"
                    )
                else:
                    no_progress_batches = 0

                if no_progress_batches >= 2:
                    logger.warning(
                        f"Aborting cleanup_expired_exports loop for user={self.user_id} "
                        "after repeated no-progress batches"
                    )
                    break

                # If we got fewer results than batch_size, we're done
                if len(results) < batch_size:
                    break

            return deleted_count
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error cleaning up expired exports: {e}")
            return 0

    def cleanup_import_orphans(
        self,
        age_threshold_hours: int = 24,
        batch_size: int = 100,
    ) -> int:
        """Clean up orphaned import files from failed, cancelled, or stale jobs.

        Handles three scenarios:
        1. Failed/cancelled jobs whose archive files still exist on disk.
        2. Jobs stuck in pending/validating for longer than *age_threshold_hours*.
        3. Files in import_dir/temp_dir with no corresponding job record.

        Args:
            age_threshold_hours: Only process jobs/files older than this many hours.
            batch_size: Number of jobs to process per batch.

        Returns:
            Number of orphaned files deleted.
        """
        try:
            cutoff = datetime.now(timezone.utc).replace(tzinfo=None)
            cutoff -= timedelta(hours=age_threshold_hours)
            cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S.%f")
            deleted_count = 0
            no_progress_batches = 0

            # --- Phase 1: clean files for terminal-state jobs ---
            terminal_statuses = (
                ImportStatus.FAILED.value,
                ImportStatus.CANCELLED.value,
                ImportStatus.COMPLETED.value,
            )
            for status_val in terminal_statuses:
                offset = 0
                while True:
                    cursor = self.db.execute_query(
                        "SELECT job_id, chatbook_path FROM import_jobs "
                        "WHERE user_id = ? AND status = ? AND created_at < ? "
                        "LIMIT ? OFFSET ?",
                        (self.user_id, status_val, cutoff_str, batch_size, offset),
                    )
                    results = self._fetch_results(cursor)
                    if not results:
                        break
                    for row in results:
                        if isinstance(row, dict):
                            chatbook_path = row.get("chatbook_path")
                        else:
                            chatbook_path = row[1] if len(row) > 1 else None

                        if chatbook_path:
                            deleted_count += self._try_delete_import_file(chatbook_path)

                    offset += len(results)
                    if len(results) < batch_size:
                        break

            # --- Phase 2: mark stale pending/validating jobs as failed ---
            stale_statuses = (
                ImportStatus.PENDING.value,
                ImportStatus.VALIDATING.value,
            )
            for status_val in stale_statuses:
                cursor = self.db.execute_query(
                    "SELECT job_id, chatbook_path FROM import_jobs "
                    "WHERE user_id = ? AND status = ? AND created_at < ? "
                    "LIMIT ?",
                    (self.user_id, status_val, cutoff_str, batch_size),
                )
                results = self._fetch_results(cursor)
                for row in (results or []):
                    if isinstance(row, dict):
                        job_id = row.get("job_id")
                        chatbook_path = row.get("chatbook_path")
                    else:
                        job_id = row[0] if row else None
                        chatbook_path = row[1] if len(row) > 1 else None

                    if chatbook_path:
                        deleted_count += self._try_delete_import_file(chatbook_path)

                    if job_id:
                        try:
                            self.db.execute_query(
                                "UPDATE import_jobs SET status = ?, error_message = ? "
                                "WHERE job_id = ?",
                                (
                                    ImportStatus.FAILED.value,
                                    "Marked failed by orphan cleanup (stale job)",
                                    job_id,
                                ),
                                commit=True,
                            )
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                            logger.warning(f"Failed to mark stale import job {job_id} as failed: {e}")

            # --- Phase 3: scan import_dir and temp_dir for untracked files ---
            for scan_dir in (self.import_dir, self.temp_dir):
                try:
                    resolved_base = Path(scan_dir).resolve()
                    if not resolved_base.is_dir():
                        continue
                    for entry in resolved_base.iterdir():
                        if not entry.is_file():
                            continue
                        try:
                            file_age = datetime.fromtimestamp(
                                entry.stat().st_mtime, tz=timezone.utc
                            ).replace(tzinfo=None)
                            if file_age > cutoff:
                                continue  # too recent
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                            continue

                        # Check if any job references this file (by absolute path or token)
                        token_ref = self._build_import_file_token(entry.resolve())
                        cursor = self.db.execute_query(
                            "SELECT 1 FROM import_jobs WHERE user_id = ? AND (chatbook_path = ? OR chatbook_path = ?) LIMIT 1",
                            (self.user_id, str(entry), token_ref),
                        )
                        refs = self._fetch_results(cursor)
                        if not refs:
                            # No job references this file — orphan
                            try:
                                entry.unlink()
                                deleted_count += 1
                                logger.debug(f"Removed orphaned import file: {entry}")
                            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                                logger.warning(f"Failed to remove orphaned import file {entry}: {e}")
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                    logger.warning(f"Error scanning {scan_dir} for orphaned imports: {e}")

            return deleted_count
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error cleaning up import orphans: {e}")
            return 0

    def _try_delete_import_file(self, file_path_str: str) -> int:
        """Try to delete an import archive file. Returns 1 if deleted, 0 otherwise."""
        try:
            file_path = self._resolve_import_archive_path(file_path_str)
        except (ValidationError, SecurityError):
            logger.warning("Refusing to delete unresolved import file token")
            return 0
        try:
            # Validate path is within expected directories
            import_base = Path(self.import_dir).resolve()
            temp_base = Path(self.temp_dir).resolve()
            in_import = os.path.commonpath([str(file_path), str(import_base)]) == str(import_base)
            in_temp = os.path.commonpath([str(file_path), str(temp_base)]) == str(temp_base)
            if not (in_import or in_temp):
                logger.warning(f"Refusing to delete import file outside expected dirs: {file_path}")
                return 0
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                return 1
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Failed to delete import file {file_path_str}: {e}")
        return 0

    def _collect_prompts(
        self,
        prompt_ids: list[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ) -> None:
        """Collect Prompt Studio prompts for export."""
        if not prompt_ids:
            return
        prompts_db = self._get_prompts_db()
        if prompts_db is None:
            logger.debug("Skipping prompt export because prompts DB is unavailable.")
            return
        prompts_dir = work_dir / "content" / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        for prompt_identifier in prompt_ids:
            prompt_record: dict[str, Any] | None = None
            # Attempt ID lookup (int) first, then UUID
            try:
                prompt_record = prompts_db.get_prompt_by_id(int(prompt_identifier))
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                prompt_record = None
            if not prompt_record:
                try:
                    prompt_record = prompts_db.get_prompt_by_uuid(str(prompt_identifier))
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                    prompt_record = None
            if not prompt_record:
                logger.debug(f"Prompt {prompt_identifier} not found; skipping.")
                continue

            prompt_payload = self._normalize_prompt_record(dict(prompt_record))
            prompt_id = str(prompt_payload.get("id", prompt_identifier))
            file_name = f"prompt_{prompt_id}.json"
            file_path = prompts_dir / file_name
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(prompt_payload, f, indent=2, ensure_ascii=False)

            content.prompts[prompt_id] = prompt_payload
            manifest.content_items.append(ContentItem(
                id=prompt_id,
                type=ContentType.PROMPT,
                title=prompt_payload.get("name", f"Prompt {prompt_id}"),
                description=prompt_payload.get("details"),
                file_path=f"content/prompts/{file_name}"
            ))

    def _collect_media_items(
        self,
        media_ids: list[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent,
        include_media: bool,
        include_embeddings: bool
    ) -> None:
        """Collect media items (metadata + transcripts) for export."""
        if not media_ids:
            return
        media_db = self._get_media_db()
        if media_db is None:
            logger.debug("Skipping media export because media DB is unavailable.")
            return

        # Apply media item cap if configured
        raw_max_items = os.getenv("CHATBOOKS_MEDIA_EXPORT_MAX_ITEMS", "0")
        try:
            max_media_items = int(raw_max_items)
        except (TypeError, ValueError):
            max_media_items = 0
        total_media_count = len(media_ids)
        if max_media_items > 0 and total_media_count > max_media_items:
            media_ids = media_ids[:max_media_items]
            truncation = manifest.truncation.setdefault("media", {})
            truncation["truncated"] = True
            truncation["max_items"] = max_media_items
            truncation["exported_count"] = max_media_items
            truncation["total_count"] = total_media_count

        media_dir = work_dir / "content" / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        embeddings_dir: Path | None = None
        binary_limits = manifest.binary_limits or {}
        emb_limit = self._resolve_binary_limit(binary_limits, "embeddings", "media_embeddings")

        if include_media:
            self._note_todo("Binary media asset export is not yet implemented; exporting metadata only.")

        for media_identifier in media_ids:
            media_record = self._fetch_media_record(media_db, str(media_identifier))
            if not media_record:
                logger.debug(f"Media {media_identifier} not found; skipping.")
                continue

            normalized = self._normalize_media_record(media_record)
            media_id = str(normalized.get("id", media_identifier))

            # Attach transcripts when helper is available
            transcripts: list[dict[str, Any]] = []
            if get_media_transcripts is not None:
                try:
                    transcripts_raw = get_media_transcripts(media_db, int(media_record["id"]))
                    transcripts = [self._normalize_transcript_row(row) for row in transcripts_raw]
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(f"Failed to fetch transcripts for media {media_id}: {exc}")
                    self._note_todo("Media transcripts export failed for some items; inspect logs.")
            normalized["transcripts"] = transcripts

            # Attach prompts linked to media when helper available
            media_prompts: list[dict[str, Any]] = []
            if get_media_prompts is not None:
                try:
                    media_prompts = get_media_prompts(media_db, int(media_record["id"]))
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(f"Failed to fetch media prompts for media {media_id}: {exc}")
                    self._note_todo("Media prompts export encountered failures; inspect logs.")
            normalized["related_prompts"] = media_prompts

            # Handle embeddings when requested and available
            vector_payload = None
            vector_blob = media_record.get("vector_embedding")
            if include_embeddings and vector_blob:
                if isinstance(vector_blob, memoryview):
                    vector_blob = vector_blob.tobytes()
                elif isinstance(vector_blob, bytearray):
                    vector_blob = bytes(vector_blob)
                if isinstance(vector_blob, (bytes, bytearray)):
                    embedding_id = f"media:{media_id}"
                    # Check binary limit before bundling
                    if emb_limit is not None and len(vector_blob) > emb_limit:
                        stub = {
                            "id": embedding_id,
                            "source": {
                                "media_id": media_id,
                                "media_uuid": normalized.get("uuid")
                            },
                            "bundled": False,
                            "size_bytes": len(vector_blob)
                        }
                        content.embeddings[embedding_id] = stub
                        manifest.content_items.append(ContentItem(
                            id=embedding_id,
                            type=ContentType.EMBEDDING,
                            title=f"Embedding for media {normalized.get('title', media_id)}",
                            metadata={"bundled": False, "size_bytes": len(vector_blob)}
                        ))
                    else:
                        embeddings_dir = embeddings_dir or (work_dir / "content" / "embeddings")
                        embeddings_dir.mkdir(parents=True, exist_ok=True)
                        vector_payload = {
                            "id": embedding_id,
                            "source": {
                                "media_id": media_id,
                                "media_uuid": normalized.get("uuid")
                            },
                            "encoding": "base64",
                            "vector": base64.b64encode(vector_blob).decode("ascii")
                        }
                        embed_file = embeddings_dir / f"embedding_media_{media_id}.json"
                        with open(embed_file, "w", encoding="utf-8") as ef:
                            json.dump(vector_payload, ef, indent=2, ensure_ascii=False)
                        content.embeddings[embedding_id] = vector_payload
                        manifest.content_items.append(ContentItem(
                            id=embedding_id,
                            type=ContentType.EMBEDDING,
                            title=f"Embedding for media {normalized.get('title', media_id)}",
                            file_path=f"content/embeddings/{embed_file.name}"
                        ))
                else:
                    self._note_todo("Encountered non-binary media vector embedding; skipping serialization.")

            media_file = media_dir / f"media_{media_id}.json"
            with open(media_file, "w", encoding="utf-8") as mf:
                json.dump(normalized, mf, indent=2, ensure_ascii=False)
            content.media[media_id] = normalized

            manifest.content_items.append(ContentItem(
                id=media_id,
                type=ContentType.MEDIA,
                title=normalized.get("title", f"Media {media_id}"),
                description=normalized.get("description"),
                file_path=f"content/media/{media_file.name}"
            ))

        if include_embeddings and not content.embeddings:
            self._note_todo("Embeddings export requested but no vector data found in media records.")

    def _collect_embeddings(
        self,
        embedding_ids: list[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ) -> None:
        """Collect ChromaDB collection-level embeddings for export."""
        chroma = self._get_chroma_manager()
        if chroma is None:
            logger.debug("Skipping embedding export because ChromaDB is unavailable.")
            return

        embeddings_dir = work_dir / "content" / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        raw_max_chunks = os.getenv("CHATBOOKS_EMBEDDING_EXPORT_MAX_CHUNKS", "10000")
        try:
            max_chunks_per_collection = int(raw_max_chunks)
        except (TypeError, ValueError):
            max_chunks_per_collection = 10000

        binary_limits = manifest.binary_limits or {}
        emb_limit = self._resolve_binary_limit(binary_limits, "embeddings", "collection_embeddings")

        # Determine which collections to export
        try:
            if embedding_ids:
                collections = []
                for name in embedding_ids:
                    try:
                        col = chroma.get_collection(collection_name=name)
                        collections.append(col)
                    except (KeyError, RuntimeError):
                        logger.debug(f"Collection '{name}' not found; skipping.")
            else:
                collections = list(chroma.list_collections())
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"Failed to list ChromaDB collections for export: {exc}")
            self._note_todo("ChromaDB collection listing failed; inspect logs.")
            return

        for collection in collections:
            try:
                col_name = collection.name
                col_metadata = collection.metadata or {}
                source_hash = hashlib.sha256(
                    json.dumps(col_metadata, sort_keys=True).encode("utf-8")
                ).hexdigest()

                total_count = collection.count()
                chunks: list[dict[str, Any]] = []
                offset = 0
                page_size = 1000
                truncated = False

                while True:
                    result = collection.get(
                        limit=page_size, offset=offset,
                        include=["documents", "metadatas", "embeddings"]
                    )
                    ids = result.get("ids", [])
                    if not ids:
                        break
                    documents = result.get("documents", [])
                    metadatas = result.get("metadatas", [])
                    embeddings_data = result.get("embeddings", [])

                    for i, chunk_id in enumerate(ids):
                        if max_chunks_per_collection > 0 and len(chunks) >= max_chunks_per_collection:
                            truncated = True
                            break
                        chunk: dict[str, Any] = {"id": chunk_id}
                        if documents and i < len(documents):
                            chunk["document"] = documents[i]
                        if metadatas and i < len(metadatas):
                            chunk["metadata"] = metadatas[i]
                        if embeddings_data and i < len(embeddings_data):
                            chunk["embedding"] = embeddings_data[i]
                        chunks.append(chunk)

                    if truncated or len(ids) < page_size:
                        break
                    offset += len(ids)

                collection_data = {
                    "embedding_set_id": col_name,
                    "source_hash": source_hash,
                    "collection_metadata": col_metadata,
                    "item_count": total_count,
                    "truncated": truncated,
                    "chunks": chunks
                }

                # Check binary limit on serialized size
                serialized = json.dumps(collection_data, ensure_ascii=False)
                serialized_size = len(serialized.encode("utf-8"))
                if emb_limit is not None and serialized_size > emb_limit:
                    stub = {
                        "embedding_set_id": col_name,
                        "source_hash": source_hash,
                        "collection_metadata": col_metadata,
                        "item_count": total_count,
                        "bundled": False,
                        "size_bytes": serialized_size
                    }
                    content.embeddings[f"collection:{col_name}"] = stub
                    manifest.content_items.append(ContentItem(
                        id=f"collection:{col_name}",
                        type=ContentType.EMBEDDING,
                        title=f"Embedding collection {col_name}",
                        metadata={"bundled": False, "size_bytes": serialized_size}
                    ))
                    trunc = manifest.truncation.setdefault("embeddings", {})
                    trunc["truncated"] = True
                    trunc.setdefault("binary_limited_collections", [])
                    if col_name not in trunc["binary_limited_collections"]:
                        trunc["binary_limited_collections"].append(col_name)
                    trunc["total_count"] = trunc.get("total_count", 0) + total_count
                    continue

                safe_name = col_name.replace("/", "_").replace("\\", "_")
                embed_file = embeddings_dir / f"collection_{safe_name}.json"
                with open(embed_file, "w", encoding="utf-8") as ef:
                    ef.write(serialized)

                content.embeddings[f"collection:{col_name}"] = collection_data
                manifest.content_items.append(ContentItem(
                    id=f"collection:{col_name}",
                    type=ContentType.EMBEDDING,
                    title=f"Embedding collection {col_name}",
                    file_path=f"content/embeddings/{embed_file.name}",
                    metadata={"source_hash": source_hash, "item_count": total_count}
                ))

                if truncated:
                    trunc = manifest.truncation.setdefault("embeddings", {})
                    trunc["truncated"] = True
                    trunc["max_chunks_per_collection"] = max_chunks_per_collection
                    col_ids = trunc.setdefault("collection_ids", [])
                    if col_name not in col_ids:
                        col_ids.append(col_name)
                    trunc["exported_count"] = trunc.get("exported_count", 0) + len(chunks)
                    trunc["total_count"] = trunc.get("total_count", 0) + total_count

            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to export collection '{collection.name}': {exc}")
                self._note_todo(f"Embedding collection export failed for '{collection.name}'; inspect logs.")

    def _collect_evaluations(
        self,
        evaluation_ids: list[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ) -> None:
        """Collect evaluation definitions and runs for export."""
        if not evaluation_ids:
            return
        evals_db = self._get_evaluations_db()
        if evals_db is None:
            logger.debug("Skipping evaluation export because evaluations DB is unavailable.")
            return
        eval_dir = work_dir / "content" / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)

        for eval_id in evaluation_ids:
            record = None
            try:
                record = evals_db.get_evaluation(str(eval_id))
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to fetch evaluation {eval_id}: {exc}")
                record = None
            if not record:
                continue

            normalized = self._normalize_evaluation_record(record)
            runs_payload: list[dict[str, Any]] = []
            try:
                raw_max_rows = os.getenv("CHATBOOKS_EVAL_EXPORT_MAX_ROWS", "200")
                try:
                    max_rows = int(raw_max_rows)
                except (TypeError, ValueError):
                    max_rows = 200
                runs, has_more = evals_db.list_runs(eval_id=str(eval_id), limit=max_rows, return_has_more=True)
                runs_payload = [self._normalize_evaluation_run(run) for run in runs]
                if has_more:
                    normalized["truncated"] = True
                    normalized["max_rows"] = max_rows
                    truncation = manifest.truncation.setdefault("evaluations", {})
                    truncation["truncated"] = True
                    truncation["max_rows"] = max_rows
                    truncation["exported_count"] = truncation.get("exported_count", 0) + len(runs_payload)
                    # total_count not knowable without a separate count query; omit
                    if runs_payload:
                        last_run_id = runs_payload[-1].get("id")
                        if last_run_id:
                            continuations = truncation.setdefault("continuations", [])
                            continuations.append({
                                "evaluation_id": str(eval_id),
                                "run_id": str(last_run_id),
                                "continuation_token": str(last_run_id)
                            })
                    self._note_todo("Evaluation export limited to max rows; add resumable export support.")
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to list evaluation runs for {eval_id}: {exc}")
                self._note_todo("Evaluation runs export failed for some items; inspect logs.")
            normalized["runs"] = runs_payload

            eval_file = eval_dir / f"evaluation_{eval_id}.json"
            with open(eval_file, "w", encoding="utf-8") as ef:
                json.dump(normalized, ef, indent=2, ensure_ascii=False)
            content.evaluations[str(eval_id)] = normalized

            manifest.content_items.append(ContentItem(
                id=str(eval_id),
                type=ContentType.EVALUATION,
                title=normalized.get("name", f"Evaluation {eval_id}"),
                description=normalized.get("description"),
                file_path=f"content/evaluations/{eval_file.name}"
            ))

    # Helper methods for collecting content

    def _collect_conversations(
        self,
        conversation_ids: list[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ):
        """Collect conversations for export."""
        conv_dir = work_dir / "content" / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        binary_limits = manifest.binary_limits or {}
        attachment_limit = self._resolve_binary_limit(
            binary_limits, "conversations", "conversation", "attachments"
        )

        for conv_id in conversation_ids:
            try:
                # Get conversation
                conv = self.db.get_conversation_by_id(conv_id)
                if not conv:
                    continue

                # Get messages (paged to avoid silent truncation)
                messages, truncated, max_messages = self._get_conversation_messages_paged(conv_id)
                if truncated:
                    truncation = manifest.truncation.setdefault("conversations", {})
                    truncation["truncated"] = True
                    truncation["max_messages"] = max_messages
                    truncation["exported_count"] = truncation.get("exported_count", 0) + len(messages or [])
                    conv_ids = truncation.setdefault("conversation_ids", [])
                    if str(conv_id) not in conv_ids:
                        conv_ids.append(str(conv_id))

                attachments_dir: Path | None = None
                conversation_messages: list[dict[str, Any]] = []
                for msg in (messages or []):
                    message_payload: dict[str, Any] = {
                        "id": msg['id'],
                        "role": msg['sender'],
                        "content": msg.get('message', msg.get('content', '')),
                        "timestamp": msg['timestamp'].isoformat() if hasattr(msg['timestamp'], 'isoformat') else msg['timestamp'],
                        "attachments": [],
                        "citations": []
                    }

                    # Persist primary image (messages.image_data) as an attachment if present
                    primary_bytes = msg.get("image_data")
                    if isinstance(primary_bytes, memoryview):
                        primary_bytes = primary_bytes.tobytes()
                    primary_mime = msg.get("image_mime_type") or "application/octet-stream"
                    if primary_bytes:
                        if attachment_limit is not None and len(primary_bytes) > attachment_limit:
                            message_payload["attachments"].append({
                                "type": "image",
                                "mime_type": primary_mime,
                                "file_path": None,
                                "bundled": False,
                                "size_bytes": len(primary_bytes),
                                "primary": True
                            })
                        else:
                            if attachments_dir is None:
                                attachments_dir = conv_dir / f"conversation_{conv_id}_assets"
                                attachments_dir.mkdir(parents=True, exist_ok=True)
                            ext = self._extension_from_mime(primary_mime)
                            attachment_name = f"{msg['id']}_image_primary{ext}"
                            attachment_path = attachments_dir / attachment_name
                            try:
                                with open(attachment_path, "wb") as af:
                                    af.write(bytes(primary_bytes))
                            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                                logger.debug(f"Failed to persist primary image attachment for message {msg['id']}: {exc}")
                                self._note_todo("Failed to export some conversation image attachments; inspect logs.")
                            else:
                                rel_path = f"content/conversations/{attachments_dir.name}/{attachment_name}"
                                message_payload["attachments"].append({
                                    "type": "image",
                                    "mime_type": primary_mime,
                                    "file_path": rel_path,
                                    "bundled": True,
                                    "primary": True
                                })

                    # Persist inline images as attachments
                    for idx, image in enumerate(msg.get("images") or []):
                        image_bytes = image.get("image_data")
                        if isinstance(image_bytes, memoryview):
                            image_bytes = image_bytes.tobytes()
                        if not image_bytes:
                            continue
                        image_mime = image.get("image_mime_type") or "application/octet-stream"
                        if primary_bytes and image_bytes == primary_bytes and image_mime == primary_mime:
                            continue
                        if attachment_limit is not None and len(image_bytes) > attachment_limit:
                            message_payload["attachments"].append({
                                "type": "image",
                                "mime_type": image_mime,
                                "file_path": None,
                                "bundled": False,
                                "size_bytes": len(image_bytes)
                            })
                            continue
                        if attachments_dir is None:
                            attachments_dir = conv_dir / f"conversation_{conv_id}_assets"
                            attachments_dir.mkdir(parents=True, exist_ok=True)
                        ext = self._extension_from_mime(image_mime)
                        attachment_name = f"{msg['id']}_image_{idx}{ext}"
                        attachment_path = attachments_dir / attachment_name
                        try:
                            with open(attachment_path, "wb") as af:
                                af.write(bytes(image_bytes))
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                            logger.debug(f"Failed to persist image attachment for message {msg['id']}: {exc}")
                            self._note_todo("Failed to export some conversation image attachments; inspect logs.")
                            continue
                        rel_path = f"content/conversations/{attachments_dir.name}/{attachment_name}"
                        message_payload["attachments"].append({
                            "type": "image",
                            "mime_type": image_mime,
                            "file_path": rel_path,
                            "bundled": True
                        })

                    # Extract citations from RAG context if available
                    try:
                        rag_context = self.db.get_message_rag_context(msg['id'])
                        if isinstance(rag_context, dict) and rag_context:
                            # Include retrieved documents as citations
                            retrieved_docs = rag_context.get('retrieved_documents', [])
                            for doc in retrieved_docs:
                                citation_entry = {
                                    "id": doc.get('id'),
                                    "source_type": doc.get('source_type'),
                                    "title": doc.get('title'),
                                    "score": doc.get('score'),
                                    "excerpt": doc.get('excerpt'),
                                    "url": doc.get('url'),
                                    "page_number": doc.get('page_number'),
                                    "chunk_id": doc.get('chunk_id'),
                                }
                                # Remove None values for cleaner export
                                citation_entry = {k: v for k, v in citation_entry.items() if v is not None}
                                if citation_entry:
                                    message_payload["citations"].append(citation_entry)

                            # Also include formal citations if present
                            formal_citations = rag_context.get('citations', [])
                            if formal_citations:
                                message_payload["formal_citations"] = formal_citations

                            # Include the RAG settings snapshot for reproducibility
                            settings_snapshot = rag_context.get('settings_snapshot')
                            if settings_snapshot:
                                message_payload["rag_settings"] = settings_snapshot

                            # Include generated answer metadata
                            if rag_context.get('generated_answer'):
                                message_payload["rag_generated_answer"] = rag_context.get('generated_answer')

                            # Include search query for context
                            if rag_context.get('search_query'):
                                message_payload["rag_search_query"] = rag_context.get('search_query')
                    except _CHATBOOK_NONCRITICAL_EXCEPTIONS as cite_err:
                        logger.debug(f"Failed to extract citations for message {msg['id']}: {cite_err}")

                    conversation_messages.append(message_payload)

                conv_data = {
                    "id": conv['id'],
                    "name": conv.get('title', 'Untitled'),
                    "created_at": conv['created_at'].isoformat() if hasattr(conv['created_at'], 'isoformat') else conv['created_at'],
                    "character_id": conv.get('character_id'),
                    "attachments_path": f"content/conversations/{attachments_dir.name}" if attachments_dir else None,
                    "messages": conversation_messages
                }

                # Write to file
                conv_file = conv_dir / f"conversation_{conv_id}.json"
                with open(conv_file, 'w', encoding='utf-8') as f:
                    json.dump(conv_data, f, indent=2, ensure_ascii=False)

                # Add to content
                content.conversations[conv_id] = conv_data

                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=conv_id,
                    type=ContentType.CONVERSATION,
                    title=conv_data['name'],
                    file_path=f"content/conversations/conversation_{conv_id}.json"
                ))

            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Error collecting conversation {conv_id}: {e}")

    def _collect_notes(
        self,
        note_ids: list[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ):
        """Collect notes for export."""
        notes_dir = work_dir / "content" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        template_settings = self._resolve_template_settings(manifest)

        def _yaml_scalar(value: Any) -> str:
            """Render a safe YAML scalar for frontmatter."""
            text = "" if value is None else str(value)
            needs_quote = False
            if text.strip() != text:
                needs_quote = True
            if any(ch in text for ch in ['\n', '\r', ':', '#', '{', '}', '[', ']', ',', '&', '*', '!', '|', '>', '%', '@', '`']):
                needs_quote = True
            if text.startswith(("-", "?", " ")):
                needs_quote = True
            if needs_quote:
                escaped = text.replace("\\", "\\\\").replace('"', '\\"')
                return f"\"{escaped}\""
            return text

        for note_id in note_ids:
            try:
                # Get note
                note = self.db.get_note_by_id(note_id)
                if not note:
                    continue

                rendered_title = self._render_chatbook_text(
                    note.get("title", ""),
                    template_settings=template_settings,
                    stage="export",
                    metrics_source="chatbook",
                )
                rendered_content = self._render_chatbook_text(
                    note.get("content", ""),
                    template_settings=template_settings,
                    stage="export",
                    metrics_source="chatbook",
                )

                # Create note data
                note_data = {
                    "id": note['id'],
                    "title": rendered_title,
                    "content": rendered_content,
                    "created_at": note['created_at'].isoformat() if hasattr(note['created_at'], 'isoformat') else note['created_at']
                }

                # Write markdown file
                note_file = notes_dir / f"note_{note_id}.md"
                with open(note_file, 'w', encoding='utf-8') as f:
                    # Write frontmatter
                    f.write("---\n")
                    f.write(f"id: {note['id']}\n")
                    f.write(f"title: {_yaml_scalar(note_data['title'])}\n")
                    f.write(f"created_at: {note_data['created_at']}\n")
                    f.write("---\n\n")
                    f.write(str(note_data['content']))

                # Add to content
                content.notes[note_id] = note_data

                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=note_id,
                    type=ContentType.NOTE,
                    title=str(note_data['title']),
                    file_path=f"content/notes/note_{note_id}.md"
                ))

            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Error collecting note {note_id}: {e}")

    def _collect_characters(
        self,
        character_ids: list[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ):
        """Collect character cards for export."""
        chars_dir = work_dir / "content" / "characters"
        chars_dir.mkdir(parents=True, exist_ok=True)

        for char_id in character_ids:
            try:
                # Get character
                char = self.db.get_character_card_by_id(int(char_id))
                if not char:
                    continue

                # Write character file
                char_file = chars_dir / f"character_{char_id}.json"
                with open(char_file, 'w', encoding='utf-8') as f:
                    json.dump(char, f, indent=2, ensure_ascii=False)

                # Add to content
                content.characters[char_id] = char

                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=char_id,
                    type=ContentType.CHARACTER,
                    title=char.get('name', 'Unnamed'),
                    file_path=f"content/characters/character_{char_id}.json"
                ))

            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Error collecting character {char_id}: {e}")

    def _collect_world_books(
        self,
        world_book_ids: list[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ):
        """Collect world books for export."""
        wb_dir = work_dir / "content" / "world_books"
        wb_dir.mkdir(parents=True, exist_ok=True)

        # Import the world book service
        from ..Character_Chat.world_book_manager import WorldBookService

        wb_service = WorldBookService(self.db)

        for wb_id in world_book_ids:
            try:
                # Get world book with entries
                wb_data = wb_service.export_world_book(int(wb_id))

                # Convert datetime objects to strings for JSON serialization
                wb_data_serializable = self._convert_datetimes(wb_data)

                # Write world book file
                wb_file = wb_dir / f"world_book_{wb_id}.json"
                with open(wb_file, 'w', encoding='utf-8') as f:
                    json.dump(wb_data_serializable, f, indent=2, ensure_ascii=False)

                # Add to content
                content.world_books[wb_id] = wb_data

                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=wb_id,
                    type=ContentType.WORLD_BOOK,
                    title=wb_data.get('name', 'Unnamed'),
                    file_path=f"content/world_books/world_book_{wb_id}.json"
                ))

            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Error collecting world book {wb_id}: {e}")

    def _collect_dictionaries(
        self,
        dictionary_ids: list[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ):
        """Collect chat dictionaries for export."""
        dict_dir = work_dir / "content" / "dictionaries"
        dict_dir.mkdir(parents=True, exist_ok=True)
        template_settings = self._resolve_template_settings(manifest)

        # Import the dictionary service
        from ..Character_Chat.chat_dictionary import ChatDictionaryService

        dict_service = ChatDictionaryService(self.db)

        for dict_id in dictionary_ids:
            try:
                # Get dictionary with entries
                dict_data = dict_service.export_to_json(int(dict_id))
                dict_meta = dict_service.get_dictionary(int(dict_id))
                if dict_meta:
                    dict_data.setdefault("name", dict_meta.get("name"))
                    dict_data.setdefault("description", dict_meta.get("description"))
                    dict_data["id"] = dict_meta.get("id", dict_id)
                    dict_data["is_active"] = dict_meta.get("is_active", True)

                dict_data["name"] = self._render_chatbook_text(
                    dict_data.get("name", "Unnamed"),
                    template_settings=template_settings,
                    stage="export",
                    metrics_source="chatbook",
                )
                dict_data["description"] = self._render_chatbook_text(
                    dict_data.get("description", ""),
                    template_settings=template_settings,
                    stage="export",
                    metrics_source="chatbook",
                )

                entries = dict_data.get("entries")
                if isinstance(entries, list):
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        if "replacement" in entry:
                            entry["replacement"] = self._render_chatbook_text(
                                entry.get("replacement"),
                                template_settings=template_settings,
                                stage="export",
                                metrics_source="dict",
                                require_dict_templates_enabled=True,
                            )
                        if "content" in entry:
                            entry["content"] = self._render_chatbook_text(
                                entry.get("content"),
                                template_settings=template_settings,
                                stage="export",
                                metrics_source="dict",
                                require_dict_templates_enabled=True,
                            )

                # Convert datetime objects to strings for JSON serialization
                dict_data_serializable = self._convert_datetimes(dict_data)

                # Write dictionary file
                dict_file = dict_dir / f"dictionary_{dict_id}.json"
                with open(dict_file, 'w', encoding='utf-8') as f:
                    json.dump(dict_data_serializable, f, indent=2, ensure_ascii=False)

                # Add to content
                content.dictionaries[dict_id] = dict_data

                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=dict_id,
                    type=ContentType.DICTIONARY,
                    title=dict_data.get('name', 'Unnamed'),
                    file_path=f"content/dictionaries/dictionary_{dict_id}.json"
                ))

            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Error collecting dictionary {dict_id}: {e}")

    def _collect_generated_documents(
        self,
        document_ids: list[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ):
        """Collect generated documents for export."""
        docs_dir = work_dir / "content" / "generated_documents"
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Import the document generator service
        from ..Chat.document_generator import DocumentGeneratorService

        doc_service = DocumentGeneratorService(self.db, self.user_id)

        for doc_id in document_ids:
            try:
                # Get document
                doc = doc_service.get_document(doc_id)
                if not doc:
                    continue

                # Write document file
                doc_file = docs_dir / f"document_{doc_id}.json"
                with open(doc_file, 'w', encoding='utf-8') as f:
                    json.dump(doc, f, indent=2, ensure_ascii=False)

                # Add to content
                content.generated_documents[doc_id] = doc

                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=doc_id,
                    type=ContentType.GENERATED_DOCUMENT,
                    title=doc.get('title', 'Untitled'),
                    file_path=f"content/generated_documents/document_{doc_id}.json"
                ))

            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Error collecting document {doc_id}: {e}")

    # Helper methods for importing content

    def _import_conversations(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        conversation_ids: list[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob,
        character_id_map: dict[str, int] | None = None
    ):
        """Import conversations from chatbook."""
        conv_dir = extract_dir / "content" / "conversations"
        max_img_bytes = self._get_max_message_image_bytes()

        for conv_id in conversation_ids:
            status.processed_items += 1

            try:
                # Load conversation file
                conv_file = conv_dir / f"conversation_{conv_id}.json"
                if not conv_file.exists():
                    status.failed_items += 1
                    status.warnings.append(f"Conversation file not found: {conv_file.name}")
                    continue

                with open(conv_file, encoding='utf-8') as f:
                    conv_data = json.load(f)

                # Check for existing conversation
                conv_name = (
                    conv_data.get('name')
                    or conv_data.get('title')
                    or conv_data.get('conversation_title')
                    or 'Untitled'
                )
                if prefix_imported:
                    conv_name = f"[Imported] {conv_name}"

                existing = self._get_conversation_by_name(conv_name)
                if existing and conflict_resolution == ConflictResolution.SKIP:
                    status.skipped_items += 1
                    continue
                elif existing and conflict_resolution == ConflictResolution.RENAME:
                    conv_name = self._generate_unique_name(conv_name, "conversation")

                resolved_char_id, warn = self._resolve_import_character_id(
                    conv_data.get('character_id'),
                    character_id_map=character_id_map,
                )
                if warn:
                    status.warnings.append(warn)
                if resolved_char_id is None:
                    status.failed_items += 1
                    status.warnings.append(f"Conversation {conv_id} skipped due to missing character_id.")
                    continue

                # Create conversation + messages atomically
                with self.db.transaction():
                    conv_dict = {
                        'title': conv_name,
                        'created_at': conv_data.get('created_at') or conv_data.get('created') or conv_data.get('timestamp'),
                        'character_id': resolved_char_id
                    }
                    new_conv_id = self.db.add_conversation(conv_dict)

                    if new_conv_id:
                        # Import messages
                        base_path = extract_dir.resolve()
                        for msg in conv_data.get('messages', []):
                            raw_role = msg.get('role') or msg.get('sender') or msg.get('author') or msg.get('from')
                            raw_content = msg.get('content')
                            if raw_content is None:
                                raw_content = msg.get('message')
                            if raw_content is None:
                                raw_content = msg.get('text')
                            msg_dict = {
                                'conversation_id': new_conv_id,
                                'sender': raw_role or 'user',
                                'content': raw_content if raw_content is not None else '',
                                'timestamp': msg.get('timestamp') or msg.get('created_at')
                            }

                            attachments = msg.get('attachments') or []
                            images_payload: list[dict[str, Any]] = []
                            primary_payload: tuple[bytes, str] | None = None
                            for attachment in attachments:
                                if not isinstance(attachment, dict):
                                    continue
                                if str(attachment.get("type", "")).lower() != "image":
                                    continue
                                rel_path = attachment.get("file_path")
                                if not rel_path:
                                    continue
                                try:
                                    attachment_rel = Path(rel_path)
                                except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                                    continue
                                candidate_path = (base_path / attachment_rel).resolve()
                                try:
                                    candidate_path.relative_to(base_path)
                                except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                                    status.warnings.append(f"Skipped attachment outside extract dir: {rel_path}")
                                    continue
                                try:
                                    size_bytes = candidate_path.stat().st_size
                                except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                                    size_bytes = None
                                if size_bytes is not None and size_bytes > max_img_bytes:
                                    status.warnings.append(
                                        f"Skipped attachment {rel_path}: exceeds MAX_MESSAGE_IMAGE_BYTES ({max_img_bytes} bytes)"
                                    )
                                    continue
                                try:
                                    image_bytes = candidate_path.read_bytes()
                                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as read_exc:
                                    status.warnings.append(f"Failed to read attachment {rel_path}: {read_exc}")
                                    continue
                                if len(image_bytes) > max_img_bytes:
                                    status.warnings.append(
                                        f"Skipped attachment {rel_path}: exceeds MAX_MESSAGE_IMAGE_BYTES ({max_img_bytes} bytes)"
                                    )
                                    continue
                                mime_type = attachment.get("mime_type") or "application/octet-stream"
                                if attachment.get("primary") or attachment.get("is_primary"):
                                    if primary_payload is None:
                                        primary_payload = (image_bytes, mime_type)
                                    else:
                                        images_payload.append({
                                            "image_data": image_bytes,
                                            "image_mime_type": mime_type
                                        })
                                else:
                                    images_payload.append({
                                        "image_data": image_bytes,
                                        "image_mime_type": mime_type
                                    })
                            if primary_payload:
                                msg_dict['image_data'] = primary_payload[0]
                                msg_dict['image_mime_type'] = primary_payload[1]
                            if images_payload:
                                msg_dict['images'] = images_payload

                            self.db.add_message(msg_dict)

                        status.successful_items += 1
                    else:
                        # If add failed, it might be a duplicate not caught by search
                        # Count as skipped if we're in skip mode
                        if conflict_resolution == ConflictResolution.SKIP:
                            status.skipped_items += 1
                        else:
                            status.failed_items += 1

            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                status.failed_items += 1
                status.warnings.append(f"Error importing conversation {conv_id}: {str(e)}")

    def _import_notes(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        note_ids: list[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob
    ):
        """Import notes from chatbook."""
        notes_dir = extract_dir / "content" / "notes"
        template_settings = self._resolve_template_settings(manifest)

        def _parse_yaml_scalar(text: str) -> str:
            if not text:
                return ""
            if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
                inner = text[1:-1]
                if text[0] == '"':
                    inner = (
                        inner.replace("\\\\", "\\")
                        .replace("\\\"", "\"")
                        .replace("\\n", "\n")
                        .replace("\\r", "\r")
                        .replace("\\t", "\t")
                    )
                else:
                    inner = inner.replace("''", "'")
                return inner
            return text
        def _extract_title_frontmatter(frontmatter: str, fallback: str) -> str:
            for line in frontmatter.splitlines():
                if line.startswith('title:'):
                    raw = line.replace('title:', '', 1).strip()
                    return _parse_yaml_scalar(raw) or fallback
            return fallback

        for note_id in note_ids:
            status.processed_items += 1

            try:
                # Find note file
                note_file = notes_dir / f"note_{note_id}.md"
                if not note_file.exists():
                    status.failed_items += 1
                    status.warnings.append(f"Note file not found: {note_file.name}")
                    continue

                # Parse markdown with frontmatter
                with open(note_file, encoding='utf-8') as f:
                    content = f.read()

                # Extract frontmatter
                note_content = content
                note_title = f"Imported Note {note_id}"

                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        # Parse frontmatter for title
                        frontmatter = parts[1].strip()
                        try:
                            import yaml  # type: ignore
                            parsed = yaml.safe_load(frontmatter) if frontmatter else {}
                            if isinstance(parsed, dict) and parsed.get('title') is not None:
                                note_title = str(parsed.get('title'))
                            else:
                                note_title = _extract_title_frontmatter(frontmatter, note_title)
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                            note_title = _extract_title_frontmatter(frontmatter, note_title)
                        note_content = parts[2].strip()

                note_title = self._render_chatbook_text(
                    note_title,
                    template_settings=template_settings,
                    stage="import",
                    metrics_source="chatbook",
                )
                note_content = self._render_chatbook_text(
                    note_content,
                    template_settings=template_settings,
                    stage="import",
                    metrics_source="chatbook",
                )

                if prefix_imported:
                    note_title = f"[Imported] {note_title}"

                # Check for existing note
                existing = self._get_note_by_title(note_title)
                if existing and conflict_resolution == ConflictResolution.SKIP:
                    status.skipped_items += 1
                    continue
                elif existing and conflict_resolution == ConflictResolution.RENAME:
                    note_title = self._generate_unique_name(note_title, "note")

                # Create note
                new_note_id = self.db.add_note(title=note_title, content=note_content)

                if new_note_id:
                    status.successful_items += 1
                else:
                    # If add failed, it might be a duplicate not caught by search
                    # Count as skipped if we're in skip mode
                    if conflict_resolution == ConflictResolution.SKIP:
                        status.skipped_items += 1
                    else:
                        status.failed_items += 1

            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                status.failed_items += 1
                status.warnings.append(f"Error importing note {note_id}: {str(e)}")

    def _import_characters(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        character_ids: list[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob,
        character_id_map: dict[str, int] | None = None
    ):
        """Import character cards from chatbook."""
        chars_dir = extract_dir / "content" / "characters"

        for char_id in character_ids:
            status.processed_items += 1

            try:
                # Load character file
                char_file = chars_dir / f"character_{char_id}.json"
                if not char_file.exists():
                    status.failed_items += 1
                    status.warnings.append(f"Character file not found: {char_file.name}")
                    continue

                with open(char_file, encoding='utf-8') as f:
                    char_data = json.load(f)

                # Check for existing character
                char_name = char_data.get('name', 'Unnamed')
                if prefix_imported:
                    char_name = f"[Imported] {char_name}"
                    char_data['name'] = char_name

                existing = self.db.get_character_card_by_name(char_name)
                if existing and conflict_resolution == ConflictResolution.SKIP:
                    status.skipped_items += 1
                    if character_id_map is not None and existing.get("id") is not None:
                        character_id_map[str(char_id)] = int(existing["id"])
                    continue
                elif existing and conflict_resolution == ConflictResolution.RENAME:
                    char_name = self._generate_unique_name(char_name, "character")
                    char_data['name'] = char_name

                # Create character
                new_char_id = self.db.add_character_card(char_data)

                if new_char_id:
                    if character_id_map is not None:
                        try:
                            character_id_map[str(char_id)] = int(new_char_id)
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                            character_id_map[str(char_id)] = new_char_id
                    status.successful_items += 1
                else:
                    # If add failed, it might be a duplicate not caught by search
                    # Count as skipped if we're in skip mode
                    if conflict_resolution == ConflictResolution.SKIP:
                        status.skipped_items += 1
                    else:
                        status.failed_items += 1

            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                status.failed_items += 1
                status.warnings.append(f"Error importing character {char_id}: {str(e)}")

    def _import_world_books(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        world_book_ids: list[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob
    ):
        """Import world books from chatbook."""
        wb_dir = extract_dir / "content" / "world_books"

        # Import the world book service
        from ..Character_Chat.world_book_manager import WorldBookService
        wb_service = WorldBookService(self.db)

        for wb_id in world_book_ids:
            status.processed_items += 1

            try:
                # Load world book file
                wb_file = wb_dir / f"world_book_{wb_id}.json"
                if not wb_file.exists():
                    status.failed_items += 1
                    status.warnings.append(f"World book file not found: {wb_file.name}")
                    continue

                with open(wb_file, encoding='utf-8') as f:
                    wb_data = json.load(f)

                # Handle import with conflict resolution
                wb_name = wb_data.get('name', 'Unnamed')
                if prefix_imported:
                    wb_name = f"[Imported] {wb_name}"
                    wb_data['name'] = wb_name

                # Check for existing world book
                existing = wb_service.get_world_book_by_name(wb_name)
                if existing and conflict_resolution == ConflictResolution.SKIP:
                    status.skipped_items += 1
                    continue
                elif existing and conflict_resolution == ConflictResolution.RENAME:
                    wb_name = self._generate_unique_name(wb_name, "world_book")
                    wb_data['name'] = wb_name

                # Import world book
                success = wb_service.import_world_book(wb_data)

                if success:
                    status.successful_items += 1
                else:
                    status.failed_items += 1

            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                status.failed_items += 1
                status.warnings.append(f"Error importing world book {wb_id}: {str(e)}")

    def _import_dictionaries(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        dictionary_ids: list[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob
    ):
        """Import chat dictionaries from chatbook."""
        dict_dir = extract_dir / "content" / "dictionaries"
        template_settings = self._resolve_template_settings(manifest)
        strict_dict_import = self._truthy_env("CHATBOOKS_IMPORT_DICT_STRICT", False)

        # Import the dictionary service
        from ..Character_Chat.chat_dictionary import ChatDictionaryService
        dict_service = ChatDictionaryService(self.db)

        for dict_id in dictionary_ids:
            status.processed_items += 1

            try:
                # Load dictionary file
                dict_file = dict_dir / f"dictionary_{dict_id}.json"
                if not dict_file.exists():
                    status.failed_items += 1
                    status.warnings.append(f"Dictionary file not found: {dict_file.name}")
                    continue

                with open(dict_file, encoding='utf-8') as f:
                    dict_data = json.load(f)

                # Handle import with conflict resolution
                dict_name = dict_data.get('name', 'Unnamed')
                dict_name = self._render_chatbook_text(
                    dict_name,
                    template_settings=template_settings,
                    stage="import",
                    metrics_source="chatbook",
                )
                dict_data['name'] = dict_name
                dict_data['description'] = self._render_chatbook_text(
                    dict_data.get('description', ''),
                    template_settings=template_settings,
                    stage="import",
                    metrics_source="chatbook",
                )

                entries = dict_data.get('entries')
                if isinstance(entries, list):
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        if "replacement" in entry:
                            entry["replacement"] = self._render_chatbook_text(
                                entry.get("replacement"),
                                template_settings=template_settings,
                                stage="import",
                                metrics_source="dict",
                                require_dict_templates_enabled=True,
                            )
                        if "content" in entry:
                            entry["content"] = self._render_chatbook_text(
                                entry.get("content"),
                                template_settings=template_settings,
                                stage="import",
                                metrics_source="dict",
                                require_dict_templates_enabled=True,
                            )

                if prefix_imported:
                    dict_name = f"[Imported] {dict_name}"
                    dict_data['name'] = dict_name

                # Check for existing dictionary
                existing = None
                get_dictionary = getattr(dict_service, "get_dictionary", None)
                if callable(get_dictionary):
                    try:
                        existing = get_dictionary(name=dict_name)
                    except TypeError:
                        # Compatibility with legacy service/test doubles that only accept positional args.
                        existing = get_dictionary(dict_name)
                if existing is None:
                    get_dictionary_by_name = getattr(dict_service, "get_dictionary_by_name", None)
                    if callable(get_dictionary_by_name):
                        existing = get_dictionary_by_name(dict_name)
                if existing and conflict_resolution == ConflictResolution.SKIP:
                    status.skipped_items += 1
                    continue
                elif existing and conflict_resolution == ConflictResolution.RENAME:
                    dict_name = self._generate_unique_name(dict_name, "dictionary")
                    dict_data['name'] = dict_name

                # Validate dictionary entries (structure + regex/template lint)
                try:
                    from ..Chat.validate_dictionary import (
                        FATAL_ERROR_CODES as _FATAL_ERROR_CODES,
                    )
                    from ..Chat.validate_dictionary import validate_dictionary as _validate_dict

                    # Normalize entries for validator shape
                    raw_entries = dict_data.get('entries') or []
                    norm_entries: list[dict[str, Any]] = []
                    for e in raw_entries:
                        if not isinstance(e, dict):
                            continue
                        typ_val = e.get('type')
                        if not typ_val:
                            typ_val = 'regex' if bool(e.get('is_regex')) else 'literal'
                        patt = e.get('pattern') or e.get('key_pattern') or e.get('key') or ''
                        repl = e.get('replacement') or e.get('content') or ''
                        prob = e.get('probability', 1.0)
                        try:
                            prob = float(prob)
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                            prob = 1.0
                        norm_entries.append({
                            'type': str(typ_val).lower(),
                            'pattern': str(patt),
                            'replacement': str(repl),
                            'probability': prob,
                            'max_replacements': int(e.get('max_replacements', 0) or 0),
                        })

                    vres = _validate_dict({'name': dict_name, 'entries': norm_entries}, schema_version=1, strict=False)
                    if vres.errors:
                        codes = sorted({err.get('code', 'unknown') for err in vres.errors})
                        fatal_codes = sorted({c for c in codes if c in _FATAL_ERROR_CODES})
                        non_fatal_codes = sorted({c for c in codes if c not in _FATAL_ERROR_CODES})
                        if fatal_codes:
                            status.warnings.append(
                                f"Dictionary '{dict_name}' validation fatal errors: {', '.join(fatal_codes)}"
                            )
                            if strict_dict_import:
                                status.skipped_items += 1
                                # Skip importing this dictionary entirely.
                                continue
                        if non_fatal_codes:
                            status.warnings.append(
                                f"Dictionary '{dict_name}' validation non-fatal errors: {', '.join(non_fatal_codes)}"
                            )
                    if vres.warnings:
                        wc = sorted({w.get('code', 'warn') for w in vres.warnings})
                        status.warnings.append(
                            f"Dictionary '{dict_name}' validation warnings: {', '.join(wc)}"
                        )
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as _ve:
                    # Non-fatal: surface as a warning and continue with import
                    status.warnings.append(f"Dictionary '{dict_name}' validation failed: {_ve}")

                # Create dictionary
                new_dict_id = dict_service.create_dictionary(
                    dict_name,
                    dict_data.get('description', ''),
                )

                if new_dict_id:
                    if not bool(dict_data.get('is_active', True)):
                        with contextlib.suppress(_CHATBOOK_NONCRITICAL_EXCEPTIONS):
                            dict_service.update_dictionary(dictionary_id=int(new_dict_id), is_active=False)
                    # Import entries
                    for entry in dict_data.get('entries', []):
                        # Support both legacy and current export shapes
                        pat = entry.get('key_pattern') or entry.get('pattern') or entry.get('key')
                        repl = entry.get('replacement') or entry.get('content')
                        if pat is None or repl is None:
                            continue
                        is_regex = bool(entry.get('is_regex')) or (str(entry.get('type', '')).lower() == 'regex')
                        # probability in DB is stored as float 0..1; accept either 0..1 or 0..100 here
                        p_raw = entry.get('probability', 1.0)
                        try:
                            pf = float(p_raw)
                            if pf > 1.0:
                                pf = max(0.0, min(1.0, pf / 100.0))
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                            pf = 1.0
                        max_rep_raw = entry.get("max_replacements", 1)
                        try:
                            max_rep_val = int(max_rep_raw)
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                            max_rep_val = 1
                        group = entry.get("group") or entry.get("group_name")
                        timed_effects = entry.get("timed_effects")
                        if not isinstance(timed_effects, dict):
                            timed_effects = None
                        dict_service.add_entry(
                            new_dict_id,
                            key=str(pat),
                            content=str(repl),
                            probability=pf,
                            group=group,
                            timed_effects=timed_effects,
                            max_replacements=max_rep_val,
                            type="regex" if is_regex else "literal",
                            enabled=bool(entry.get("enabled", True)),
                            case_sensitive=bool(entry.get("case_sensitive", True)),
                        )
                    status.successful_items += 1
                else:
                    status.failed_items += 1

            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                status.failed_items += 1
                status.warnings.append(f"Error importing dictionary {dict_id}: {str(e)}")

    # Database helper methods

    def _save_export_job(self, job: ExportJob):
        """Save export job to database.

        Note: Uses execute_query with commit=True which handles its own transaction.
        Previous _with_transaction wrapper was removed because it created a separate
        connection that didn't share the transaction with execute_query's connection.
        """
        try:
            self.db.execute_query("""
                INSERT OR REPLACE INTO export_jobs (
                    job_id, user_id, status, chatbook_name, output_path,
                    created_at, started_at, completed_at, error_message,
                    progress_percentage, total_items, processed_items,
                    file_size_bytes, download_url, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id, job.user_id, job.status.value, job.chatbook_name,
                job.output_path,
                job.created_at.strftime('%Y-%m-%d %H:%M:%S.%f') if job.created_at else None,
                job.started_at.strftime('%Y-%m-%d %H:%M:%S.%f') if job.started_at else None,
                job.completed_at.strftime('%Y-%m-%d %H:%M:%S.%f') if job.completed_at else None,
                job.error_message, job.progress_percentage, job.total_items,
                job.processed_items, job.file_size_bytes, job.download_url,
                job.expires_at.strftime('%Y-%m-%d %H:%M:%S.%f') if job.expires_at else None
            ), commit=True)
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error saving export job: {e}")
            raise

    def _claim_export_job(self, job_id: str) -> bool:
        """
        Atomically claim an export job by updating its status from PENDING to IN_PROGRESS.

        This prevents race conditions where multiple workers could process the same job.

        Args:
            job_id: The export job ID to claim

        Returns:
            True if the job was successfully claimed, False if already claimed or not found
        """
        try:
            started_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')
            cursor = self.db.execute_query(
                """UPDATE export_jobs
                   SET status = ?, started_at = ?
                   WHERE job_id = ? AND user_id = ? AND status = ?""",
                (ExportStatus.IN_PROGRESS.value, started_at, job_id, self.user_id, ExportStatus.PENDING.value),
                commit=True
            )
            # Check if any row was actually updated
            rows_affected = cursor.rowcount if hasattr(cursor, 'rowcount') else 0
            if rows_affected > 0:
                logger.debug(f"Successfully claimed export job {job_id}")
                return True
            else:
                logger.debug(f"Export job {job_id} was already claimed or not found")
                return False
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error claiming export job {job_id}: {e}")
            return False

    def _claim_import_job(self, job_id: str) -> bool:
        """
        Atomically claim an import job by updating its status from PENDING to IN_PROGRESS.

        This prevents race conditions where multiple workers could process the same job.

        Args:
            job_id: The import job ID to claim

        Returns:
            True if the job was successfully claimed, False if already claimed or not found
        """
        try:
            started_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')
            cursor = self.db.execute_query(
                """UPDATE import_jobs
                   SET status = ?, started_at = ?
                   WHERE job_id = ? AND user_id = ? AND status = ?""",
                (ImportStatus.IN_PROGRESS.value, started_at, job_id, self.user_id, ImportStatus.PENDING.value),
                commit=True
            )
            # Check if any row was actually updated
            rows_affected = cursor.rowcount if hasattr(cursor, 'rowcount') else 0
            if rows_affected > 0:
                logger.debug(f"Successfully claimed import job {job_id}")
                return True
            else:
                logger.debug(f"Import job {job_id} was already claimed or not found")
                return False
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error claiming import job {job_id}: {e}")
            return False

    def _get_export_job(self, job_id: str) -> ExportJob | None:
        """Get export job from database."""
        try:
            cursor = self.db.execute_query(
                "SELECT * FROM export_jobs WHERE job_id = ? AND user_id = ?",
                (job_id, self.user_id)
            )

            # Fetch results from cursor
            results = self._fetch_results(cursor)

            if not results:
                return None

            row = results[0]
            logger.debug(f"Retrieved row type: {type(row)}, content: {row}")

            # Handle both dict (from real DB) and tuple (from mocked tests)
            if isinstance(row, tuple):
                # Convert tuple to dict using expected field order.
                # Column 13 in tests may be legacy metadata JSON; in DB it's download_url.
                col13 = row[13] if len(row) > 13 else None
                is_json_like = isinstance(col13, str) and col13.strip().startswith('{')
                row = {
                    'job_id': row[0],
                    'user_id': row[1],
                    'status': row[2],
                    'chatbook_name': row[3],
                    'output_path': row[4],
                    'created_at': row[5],
                    'started_at': row[6],
                    'completed_at': row[7],
                    'error_message': row[8] if len(row) > 8 else None,
                    'progress_percentage': row[9] if len(row) > 9 else 0,
                    'total_items': row[10] if len(row) > 10 else 0,
                    'processed_items': row[11] if len(row) > 11 else 0,
                    'file_size_bytes': row[12] if len(row) > 12 else None,
                    'download_url': None if is_json_like else (col13 if len(row) > 13 else None),
                    'metadata': col13 if is_json_like else None,
                    'expires_at': row[14] if len(row) > 14 else None
                }

            # Parse metadata if it's a JSON string
            metadata = {}
            if 'metadata' in row and row['metadata']:
                if isinstance(row['metadata'], str):
                    try:
                        metadata = json.loads(row['metadata'])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse metadata JSON: {row['metadata']}")
                elif isinstance(row['metadata'], dict):
                    metadata = row['metadata']

            return ExportJob(
                job_id=row['job_id'],
                user_id=row['user_id'],
                status=ExportStatus(row['status']),
                chatbook_name=row['chatbook_name'],
                output_path=row['output_path'],
                created_at=self._parse_timestamp(row['created_at']),
                started_at=self._parse_timestamp(row['started_at']),
                completed_at=self._parse_timestamp(row['completed_at']),
                error_message=row['error_message'],
                progress_percentage=row['progress_percentage'] or 0,
                total_items=row['total_items'] or 0,
                processed_items=row['processed_items'] or 0,
                file_size_bytes=row['file_size_bytes'],
                download_url=row.get('download_url'),
                expires_at=self._parse_timestamp(row.get('expires_at')),
                metadata=metadata
            )
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error getting export job: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _save_import_job(self, job: ImportJob):
        """Save import job to database.

        Note: Uses execute_query with commit=True which handles its own transaction.
        Previous _with_transaction wrapper was removed because it created a separate
        connection that didn't share the transaction with execute_query's connection.
        """
        try:
            self.db.execute_query("""
                INSERT OR REPLACE INTO import_jobs (
                    job_id, user_id, status, chatbook_path,
                    created_at, started_at, completed_at, error_message,
                    progress_percentage, total_items, processed_items,
                    successful_items, failed_items, skipped_items,
                    conflicts, warnings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id, job.user_id, job.status.value, job.chatbook_path,
                job.created_at.strftime('%Y-%m-%d %H:%M:%S.%f') if job.created_at else None,
                job.started_at.strftime('%Y-%m-%d %H:%M:%S.%f') if job.started_at else None,
                job.completed_at.strftime('%Y-%m-%d %H:%M:%S.%f') if job.completed_at else None,
                job.error_message, job.progress_percentage, job.total_items,
                job.processed_items, job.successful_items, job.failed_items,
                job.skipped_items, json.dumps(job.conflicts), json.dumps(job.warnings)
            ), commit=True)
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error saving import job: {e}")
            raise

    def _get_import_job(self, job_id: str) -> ImportJob | None:
        """Get import job from database."""
        try:
            cursor = self.db.execute_query(
                "SELECT * FROM import_jobs WHERE job_id = ? AND user_id = ?",
                (job_id, self.user_id)
            )

            # Fetch results from cursor
            results = self._fetch_results(cursor)

            if not results:
                return None

            row = results[0]

            # Handle both dict (from real DB) and tuple (from mocked tests)
            if isinstance(row, tuple):
                # Convert tuple to dict using expected field order
                row = {
                    'job_id': row[0],
                    'user_id': row[1],
                    'status': row[2],
                    'chatbook_path': row[3],
                    'created_at': row[4],
                    'started_at': row[5],
                    'completed_at': row[6],
                    'error_message': row[7] if len(row) > 7 else None,
                    'progress_percentage': row[8] if len(row) > 8 else 0,
                    'total_items': row[9] if len(row) > 9 else 0,
                    'processed_items': row[10] if len(row) > 10 else 0,
                    'successful_items': row[11] if len(row) > 11 else 0,
                    'failed_items': row[12] if len(row) > 12 else 0,
                    'skipped_items': row[13] if len(row) > 13 else 0,
                    'conflicts': row[14] if len(row) > 14 else '[]',
                    'warnings': row[15] if len(row) > 15 else '[]'
                }

            return ImportJob(
                job_id=row['job_id'],
                user_id=row['user_id'],
                status=ImportStatus(row['status']),
                chatbook_path=row['chatbook_path'],
                created_at=self._parse_timestamp(row['created_at']),
                started_at=self._parse_timestamp(row['started_at']),
                completed_at=self._parse_timestamp(row['completed_at']),
                error_message=row['error_message'],
                progress_percentage=row['progress_percentage'] or 0,
                total_items=row['total_items'] or 0,
                processed_items=row['processed_items'] or 0,
                successful_items=row['successful_items'] or 0,
                failed_items=row['failed_items'] or 0,
                skipped_items=row['skipped_items'] or 0,
                conflicts=json.loads(row['conflicts']) if row['conflicts'] else [],
                warnings=json.loads(row['warnings']) if row['warnings'] else []
            )
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error getting import job: {e}")
            return None

    def _generate_unique_name(self, base_name: str, item_type: str) -> str:
        """Generate a unique name for an item.

        Args:
            base_name: The base name to make unique
            item_type: Type of item (conversation, note, character, world_book, dictionary)

        Returns:
            A unique name based on the base_name

        Raises:
            ValueError: If item_type is unknown or max iterations exceeded
        """
        MAX_ITERATIONS = 1000  # Prevent infinite loops
        valid_types = {"conversation", "note", "character", "world_book", "dictionary"}

        if item_type not in valid_types:
            raise ValueError(f"Unknown item_type '{item_type}'. Valid types: {valid_types}")

        counter = 1
        while counter <= MAX_ITERATIONS:
            new_name = f"{base_name} ({counter})"

            # Check if name exists based on item type
            if item_type == "conversation":
                if not self._get_conversation_by_name(new_name):
                    return new_name
            elif item_type == "note":
                if not self._get_note_by_title(new_name):
                    return new_name
            elif item_type == "character":
                if not self.db.get_character_card_by_name(new_name):
                    return new_name
            elif item_type == "world_book":
                # Check in world books table
                result = self.db.execute_query(
                    "SELECT id FROM world_books WHERE name = ?",
                    (new_name,)
                )
                rows = self._fetch_results(result) if result is not None else []
                if not rows:
                    return new_name
            elif item_type == "dictionary":
                # Check in dictionaries table
                result = self.db.execute_query(
                    "SELECT id FROM chat_dictionaries WHERE name = ?",
                    (new_name,)
                )
                rows = self._fetch_results(result) if result is not None else []
                if not rows:
                    return new_name

            counter += 1

        # If we've exhausted iterations, raise an error
        raise ValueError(f"Could not generate unique name for '{base_name}' after {MAX_ITERATIONS} attempts")

    # Additional methods for test compatibility

    def create_export_job(self, name: str, description: str, content_types: list[str]) -> dict[str, Any]:
        """
        Create an export job (synchronous wrapper for tests).

        Args:
            name: Export name
            description: Export description
            content_types: Content types to export

        Returns:
            Job information dictionary
        """
        try:
            job_id = str(uuid4())
            job = ExportJob(
                job_id=job_id,
                user_id=self.user_id,
                status=ExportStatus.PENDING,
                chatbook_name=name,
                created_at=datetime.now(timezone.utc)
            )

            self._save_export_job(job)

            # Audit is performed at the API layer.

            return {
                "job_id": job_id,
                "status": "pending",
                "name": name,
                "description": description
            }
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            raise JobError(f"Failed to create export job: {e}", job_type="export", cause=e) from e

    def get_export_job_status(self, job_id: str) -> dict[str, Any]:
        """Get export job status."""
        job = self.get_export_job(job_id)
        if not job:
            raise JobError(f"Export job {job_id} not found", job_id=job_id)

        result = job.to_dict()
        # Ensure status is a string value
        if hasattr(job.status, 'value'):
            result["status"] = job.status.value

        # Add computed fields
        result["file_path"] = job.output_path
        result["chatbook_name"] = job.chatbook_name

        # Add content summary if available
        if job.metadata:
            result["content_summary"] = job.metadata.get("content_summary", {})
            # Handle legacy format - if content counts are at root level
            if "conversation_count" in job.metadata:
                result["content_summary"]["conversations"] = job.metadata.get("conversation_count", 0)
            if "note_count" in job.metadata:
                result["content_summary"]["notes"] = job.metadata.get("note_count", 0)
            if "character_count" in job.metadata:
                result["content_summary"]["characters"] = job.metadata.get("character_count", 0)
        else:
            result["content_summary"] = {}

        return result

    def cancel_export_job(self, job_id: str) -> bool:
        """Cancel an export job."""
        job = self._get_export_job(job_id)
        if not job:
            raise JobError(f"Export job {job_id} not found", job_id=job_id)

        if job.status in [ExportStatus.COMPLETED, ExportStatus.FAILED]:
            return False

        job.status = ExportStatus.CANCELLED
        self._save_export_job(job)
        # Best-effort cancel of in-process task
        task = self._tasks.pop(job_id, None)
        if task:
            with contextlib.suppress(_CHATBOOK_NONCRITICAL_EXCEPTIONS):
                task.cancel()
        # PS backend cancel
        if getattr(self, "_jobs_backend", "core") == "prompt_studio" and getattr(self, "_ps_job_adapter", None) is not None:
            with contextlib.suppress(_CHATBOOK_NONCRITICAL_EXCEPTIONS):
                self._ps_job_adapter.cancel(int(job_id))
        # Core backend: cancel queued or request cancel for processing in core Jobs
        if getattr(self, "_jobs_backend", "core") == "core":
            try:
                from tldw_Server_API.app.core.Jobs.manager import JobManager
                jm = getattr(self, "_core_jobs", None) or JobManager()
                # scan recent jobs for this user and domain
                for st in ("queued", "processing"):
                    jobs = jm.list_jobs(domain="chatbooks", queue="default", status=st, owner_user_id=self.user_id, limit=50)
                    for j in jobs:
                        try:
                            payload = j.get("payload") or {}
                            job_uuid = str(j.get("uuid") or j.get("id"))
                            if payload.get("chatbooks_job_id") == job_id or job_uuid == job_id:
                                jm.cancel_job(int(j["id"]))
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                            pass
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                pass

        # Audit is performed at the API layer.

        return True

    def cancel_import_job(self, job_id: str) -> bool:
        """Cancel an import job."""
        job = self._get_import_job(job_id)
        if not job:
            raise JobError(f"Import job {job_id} not found", job_id=job_id)
        if job.status in [ImportStatus.COMPLETED, ImportStatus.FAILED]:
            return False
        job.status = ImportStatus.CANCELLED
        self._save_import_job(job)
        task = self._tasks.pop(job_id, None)
        if task:
            with contextlib.suppress(_CHATBOOK_NONCRITICAL_EXCEPTIONS):
                task.cancel()
        if getattr(self, "_jobs_backend", "core") == "prompt_studio" and getattr(self, "_ps_job_adapter", None) is not None:
            with contextlib.suppress(_CHATBOOK_NONCRITICAL_EXCEPTIONS):
                self._ps_job_adapter.cancel(int(job_id))
        if getattr(self, "_jobs_backend", "core") == "core":
            try:
                from tldw_Server_API.app.core.Jobs.manager import JobManager
                jm = getattr(self, "_core_jobs", None) or JobManager()
                for st in ("queued", "processing"):
                    jobs = jm.list_jobs(domain="chatbooks", queue="default", status=st, owner_user_id=self.user_id, limit=50)
                    for j in jobs:
                        try:
                            payload = j.get("payload") or {}
                            job_uuid = str(j.get("uuid") or j.get("id"))
                            if payload.get("chatbooks_job_id") == job_id or job_uuid == job_id:
                                jm.cancel_job(int(j["id"]))
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                            pass
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                pass
        return True

    def delete_export_job(self, job_id: str, delete_file: bool = True) -> bool:
        """Remove a completed/cancelled export job record and optionally its file."""
        job = self._get_export_job(job_id)
        if not job:
            raise JobError(f"Export job {job_id} not found", job_id=job_id)

        allowed_statuses = {ExportStatus.COMPLETED, ExportStatus.CANCELLED, ExportStatus.EXPIRED, ExportStatus.FAILED}
        if job.status not in allowed_statuses:
            return False

        if delete_file and job.output_path:
            try:
                file_path = Path(job.output_path).resolve()
                expected_base = Path(self.export_dir).resolve()
                if os.path.commonpath([str(file_path), str(expected_base)]) != str(expected_base):
                    logger.warning(f"Refusing to delete export outside export dir: {file_path}")
                elif file_path.exists() and file_path.is_file():
                    file_path.unlink()
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"Failed to delete export file for job {job_id}: {exc}")

        try:
            self.db.execute_query(
                "DELETE FROM export_jobs WHERE job_id = ? AND user_id = ?",
                (job_id, self.user_id),
                commit=True,
            )
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
            logger.error(f"Failed to delete export job {job_id}: {exc}")
            raise

        return True

    def delete_import_job(self, job_id: str) -> bool:
        """Remove a completed/cancelled import job record."""
        job = self._get_import_job(job_id)
        if not job:
            raise JobError(f"Import job {job_id} not found", job_id=job_id)

        allowed_statuses = {ImportStatus.COMPLETED, ImportStatus.CANCELLED, ImportStatus.FAILED}
        if job.status not in allowed_statuses:
            return False

        try:
            self.db.execute_query(
                "DELETE FROM import_jobs WHERE job_id = ? AND user_id = ?",
                (job_id, self.user_id),
                commit=True,
            )
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
            logger.error(f"Failed to delete import job {job_id}: {exc}")
            raise

        return True

    def create_import_job(self, file_path: str, conflict_strategy: str = "skip") -> dict[str, Any]:
        """
        Create an import job (synchronous wrapper for tests).

        Args:
            file_path: Path to import file
            conflict_strategy: How to handle conflicts

        Returns:
            Job information dictionary
        """
        try:
            job_id = str(uuid4())
            job = ImportJob(
                job_id=job_id,
                user_id=self.user_id,
                status=ImportStatus.PENDING,
                chatbook_path=file_path,
                created_at=datetime.now(timezone.utc)
            )

            self._save_import_job(job)

            return {
                "job_id": job_id,
                "status": "pending",
                "file_path": file_path
            }
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            raise JobError(f"Failed to create import job: {e}", job_type="import", cause=e) from e

    def get_import_job_status(self, job_id: str) -> dict[str, Any]:
        """Get import job status."""
        job = self.get_import_job(job_id)
        if not job:
            raise JobError(f"Import job {job_id} not found", job_id=job_id)

        result = job.to_dict()
        # Ensure status is a string value
        if hasattr(job.status, 'value'):
            result["status"] = job.status.value

        # Add compatibility fields
        result["items_imported"] = job.successful_items
        result["error"] = job.error_message
        result["progress"] = job.progress_percentage
        result["conflicts_found"] = job.skipped_items  # Assuming skipped items are conflicts
        result["conflicts_resolved"] = {
            "skipped": job.skipped_items,
            "replaced": 0,
            "renamed": 0
        }

        return result

    def preview_export(self, content_types: list[str]) -> dict[str, Any]:
        """
        Preview what would be exported.

        Args:
            content_types: Types of content to preview

        Returns:
            Preview information with counts
        """
        try:
            result = {}

            # Initialize all content types to 0
            for ct in ["conversations", "characters", "world_books", "dictionaries", "notes", "prompts"]:
                result[ct] = 0

            # Get actual counts for requested types
            for content_type in content_types:
                try:
                    if content_type == "conversations":
                        cursor = self.db.execute_query(
                            "SELECT id FROM conversations WHERE deleted = 0",
                            ()
                        )
                        items = self._fetch_results(cursor)
                        result["conversations"] = len(items) if items else 0
                    elif content_type == "characters":
                        cursor = self.db.execute_query(
                            "SELECT id FROM character_cards WHERE deleted = 0",
                            ()
                        )
                        items = self._fetch_results(cursor)
                        result["characters"] = len(items) if items else 0
                    elif content_type == "notes":
                        cursor = self.db.execute_query(
                            "SELECT id FROM notes WHERE deleted = 0",
                            ()
                        )
                        items = self._fetch_results(cursor)
                        result["notes"] = len(items) if items else 0
                    elif content_type == "world_books":
                        # Try without user_id first
                        try:
                            cursor = self.db.execute_query(
                                "SELECT id FROM world_books WHERE deleted = 0",
                                ()
                            )
                            items = self._fetch_results(cursor)
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as q_err:
                            # Table might not exist or have different schema
                            logger.debug(f"world_books count query failed (no user filter): error={q_err}")
                            items = []
                        result["world_books"] = len(items) if items else 0
                    elif content_type == "dictionaries":
                        # Try to get dictionaries
                        try:
                            cursor = self.db.execute_query(
                                "SELECT id FROM chat_dictionaries WHERE deleted = 0",
                                ()
                            )
                            items = self._fetch_results(cursor)
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as q_err:
                            # Table might not exist
                            logger.debug(f"dictionaries count query failed: error={q_err}")
                            items = []
                        result["dictionaries"] = len(items) if items else 0
                    elif content_type == "prompts":
                        # Try to get prompts
                        try:
                            cursor = self.db.execute_query(
                                "SELECT id FROM prompts WHERE deleted = 0",
                                ()
                            )
                            items = self._fetch_results(cursor)
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as q_err:
                            # Table might not exist
                            logger.debug(f"prompts count query failed: error={q_err}")
                            items = []
                        result["prompts"] = len(items) if items else 0
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                    # If query fails for any type, just set to 0
                    logger.debug(f"Query failed for {content_type}: {e}")
                    result[content_type] = 0

            return result
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            raise DatabaseError(f"Failed to preview export: {e}", cause=e) from e

    def clean_old_exports(self, days_old: int = 7) -> int:
        """
        Clean up old export files.

        Args:
            days_old: Delete exports older than this many days

        Returns:
            Number of files deleted
        """
        try:
            deleted_count = 0
            cutoff_date = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_old)
            cutoff_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S.%f')

            # Query database for old exports
            cursor = self.db.execute_query(
                "SELECT job_id, output_path FROM export_jobs WHERE user_id = ? AND created_at < ?",
                (self.user_id, cutoff_str)
            )

            # Fetch results from cursor
            results = self._fetch_results(cursor)

            if results:
                for row in results:
                    # Handle both tuple and dict formats
                    if isinstance(row, dict):
                        job_id = row['job_id']
                        output_path = row['output_path']
                    else:
                        job_id = row[0] if len(row) > 0 else None
                        output_path = row[1] if len(row) > 1 else None

                    if output_path:
                        try:
                            file_path = Path(output_path).resolve()
                            expected_base = Path(self.export_dir).resolve()
                            if os.path.commonpath([str(file_path), str(expected_base)]) != str(expected_base):
                                logger.warning(f"Refusing to delete export outside export dir: {file_path}")
                            elif file_path.exists() and file_path.is_file():
                                file_path.unlink()
                                deleted_count += 1
                                logger.info(f"Deleted old export: {output_path}")
                        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                            logger.error(f"Failed to delete {output_path}: {e}")

                    # Delete from database
                    try:
                        self.db.execute_query(
                            "DELETE FROM export_jobs WHERE job_id = ?",
                            (job_id,),
                            commit=True
                        )
                    except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                        logger.error(f"Failed to delete job record {job_id}: {e}")

            # Audit is performed at the API layer.

            return deleted_count
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            raise FileOperationError(f"Failed to clean old exports: {e}", operation="cleanup", cause=e) from e

    def validate_chatbook(self, file_path: str) -> bool:
        """
        Validate a chatbook file.

        Args:
            file_path: Path to chatbook file

        Returns:
            True if valid
        """
        try:
            from .chatbook_validators import ChatbookValidator

            try:
                resolved_path = self._resolve_import_archive_path(file_path)
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                raise ValidationError("Invalid or potentially malicious archive file", field="file_path") from exc
            file_path = str(resolved_path)

            valid_archive, archive_error = ChatbookValidator.validate_zip_file(file_path)
            if not valid_archive:
                raise ValidationError(
                    archive_error or "Invalid chatbook archive",
                    field="file_path",
                )

            with zipfile.ZipFile(file_path, 'r') as zf:
                # Check for manifest
                if 'manifest.json' not in zf.namelist():
                    raise ValidationError("Missing manifest.json", field="manifest")

                # Validate manifest structure
                manifest_data = zf.read('manifest.json')
                manifest = json.loads(manifest_data)

                # Check required fields
                required_fields = ['version', 'name', 'description']
                for field in required_fields:
                    if field not in manifest:
                        raise ValidationError(f"Missing required field: {field}", field=field)

                return True
        except zipfile.BadZipFile:
            raise ArchiveError("Invalid ZIP file", archive_path=file_path) from None
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            if isinstance(e, (ValidationError, ArchiveError)):
                raise
            raise ValidationError(f"Validation failed: {e}", cause=e) from e

    def validate_chatbook_file(self, file_path: str) -> dict[str, Any]:
        """
        Validate a chatbook file (test compatibility method).

        Args:
            file_path: Path to chatbook file

        Returns:
            Dict with validation results
        """
        try:
            # Try to validate using the main method
            is_valid = self.validate_chatbook(file_path)

            # If valid, try to get manifest
            manifest = None
            if is_valid:
                try:
                    resolved_path = self._resolve_import_archive_path(file_path)
                    with zipfile.ZipFile(resolved_path, 'r') as zf:
                        manifest_data = zf.read('manifest.json')
                        manifest = json.loads(manifest_data)
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as mf_err:
                    logger.debug(f"Failed to read chatbook manifest.json: path={file_path}, error={mf_err}")

            return {
                "is_valid": is_valid,
                "manifest": manifest,
                "error": None
            }
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            return {
                "is_valid": False,
                "manifest": None,
                "error": str(e)
            }

    def get_statistics(self) -> dict[str, Any]:
        """Get import/export statistics."""
        try:
            # Get export stats
            export_cursor = self.db.execute_query(
                "SELECT status, COUNT(*) as count FROM export_jobs WHERE user_id = ? GROUP BY status",
                (self.user_id,)
            )
            export_results = self._fetch_results(export_cursor)

            # Get import stats
            import_cursor = self.db.execute_query(
                "SELECT status, COUNT(*) as count FROM import_jobs WHERE user_id = ? GROUP BY status",
                (self.user_id,)
            )
            import_results = self._fetch_results(import_cursor)

            # Build stats dict - handle both dict and tuple formats
            export_stats = {}
            for row in (export_results or []):
                if isinstance(row, dict):
                    export_stats[row["status"]] = row["count"]
                else:
                    # Tuple format (status, count)
                    export_stats[row[0]] = row[1]

            import_stats = {}
            for row in (import_results or []):
                if isinstance(row, dict):
                    import_stats[row["status"]] = row["count"]
                else:
                    # Tuple format (status, count)
                    import_stats[row[0]] = row[1]

            return {
                "exports": export_stats,
                "imports": import_stats,
                "total_exports": sum(export_stats.values()),
                "total_imports": sum(import_stats.values())
            }
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                "exports": {},
                "imports": {},
                "total_exports": 0,
                "total_imports": 0
            }

    # Removed legacy JobQueueShim handlers; Chatbooks uses in-process tasks (core) or PS adapter (prompt_studio).

    def _create_chatbook_archive(self, work_dir: Path, output_path: Path) -> bool:
        """Create ZIP archive from work directory."""
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in work_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(work_dir)
                        if arcname.as_posix() == "manifest.json":
                            zf.write(file_path, arcname, compress_type=zipfile.ZIP_STORED)
                        else:
                            zf.write(file_path, arcname)
            return True
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to create archive: {e}")
            return False

    def _write_content_to_archive(self, zf: zipfile.ZipFile, content_items: list[ContentItem], base_dir: str = "content"):
        """Write content items to archive."""
        for item in content_items:
            # Create item directory
            item_dir = f"{base_dir}/{item.type.value}/{item.id}"

            # Write item metadata
            metadata = item.to_dict()
            zf.writestr(f"{item_dir}/metadata.json", json.dumps(metadata, indent=2))

            # Write content if available
            if item.metadata:
                zf.writestr(f"{item_dir}/content.json", json.dumps(item.metadata, indent=2))

    def _process_import_items(self, items: list[ContentItem], conflict_resolution: str = "skip") -> ImportStatusData:
        """Process import items with conflict resolution."""
        status = ImportStatusData()
        status.total_items = len(items)

        for item in items:
            try:
                # Check for conflicts
                existing = None
                if item.type == ContentType.CONVERSATION:
                    existing = self.db.execute_query(
                        "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
                        (item.id, self.user_id)
                    )

                if existing and conflict_resolution == "skip":
                    status.skipped_items += 1
                    status.conflicts.append({"item_id": item.id, "action": "skipped"})
                elif existing and conflict_resolution == "overwrite":
                    # Overwrite existing
                    status.successful_items += 1
                    status.conflicts.append({"item_id": item.id, "action": "overwritten"})
                else:
                    # Import new item
                    status.successful_items += 1
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
                status.failed_items += 1
                status.warnings.append(f"Failed to import {item.id}: {str(e)}")

        return status

    async def _create_readme_async(self, work_dir: Path, manifest: ChatbookManifest):
        """Create README file for the chatbook asynchronously."""
        readme_path = work_dir / "README.md"

        content = []
        content.append(f"# {manifest.name}\n\n")
        content.append(f"{manifest.description}\n\n")

        if manifest.author:
            content.append(f"**Author:** {manifest.author}\n\n")

        content.append(f"**Created:** {manifest.created_at.strftime('%Y-%m-%d %H:%M')}\n\n")
        content.append("## Contents\n\n")

        if manifest.total_conversations > 0:
            content.append(f"- **Conversations:** {manifest.total_conversations}\n")
        if manifest.total_notes > 0:
            content.append(f"- **Notes:** {manifest.total_notes}\n")
        if manifest.total_characters > 0:
            content.append(f"- **Characters:** {manifest.total_characters}\n")
        if manifest.total_world_books > 0:
            content.append(f"- **World Books:** {manifest.total_world_books}\n")
        if manifest.total_dictionaries > 0:
            content.append(f"- **Dictionaries:** {manifest.total_dictionaries}\n")
        if manifest.total_documents > 0:
            content.append(f"- **Generated Documents:** {manifest.total_documents}\n")

        if manifest.tags:
            content.append(f"\n## Tags\n\n{', '.join(manifest.tags)}\n")

        content.append("\n## License\n\n")
        content.append(manifest.license or "See individual content files for licensing information.")

        async with aiofiles.open(readme_path, 'w', encoding='utf-8') as f:
            await f.write(''.join(content))

    def _create_readme(self, work_dir: Path, manifest: ChatbookManifest):
        """Create README file for the chatbook (sync version for backwards compatibility)."""
        readme_path = work_dir / "README.md"

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# {manifest.name}\n\n")
            f.write(f"{manifest.description}\n\n")

            if manifest.author:
                f.write(f"**Author:** {manifest.author}\n\n")

            f.write(f"**Created:** {manifest.created_at.strftime('%Y-%m-%d %H:%M')}\n\n")

            f.write("## Contents\n\n")

            if manifest.total_conversations > 0:
                f.write(f"- **Conversations:** {manifest.total_conversations}\n")
            if manifest.total_notes > 0:
                f.write(f"- **Notes:** {manifest.total_notes}\n")
            if manifest.total_characters > 0:
                f.write(f"- **Characters:** {manifest.total_characters}\n")
            if manifest.total_world_books > 0:
                f.write(f"- **World Books:** {manifest.total_world_books}\n")
            if manifest.total_dictionaries > 0:
                f.write(f"- **Dictionaries:** {manifest.total_dictionaries}\n")
            if manifest.total_documents > 0:
                f.write(f"- **Generated Documents:** {manifest.total_documents}\n")

            if manifest.tags:
                f.write(f"\n## Tags\n\n{', '.join(manifest.tags)}\n")

            f.write("\n## License\n\n")
            f.write(manifest.license or "See individual content files for licensing information.")

    async def _create_zip_archive_async(self, work_dir: Path, output_path: Path):
        """Create ZIP archive of the chatbook asynchronously with compression limits."""
        def _create_archive():
            """Write the ZIP archive, enforcing per-file and total size limits."""
            per_file_limit, total_limit = self._get_archive_limits()
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                total_size = 0
                for file_path in work_dir.rglob('*'):
                    if file_path.is_file():
                        # Check individual file size
                        file_size = file_path.stat().st_size
                        if file_size > per_file_limit:
                            max_mb = per_file_limit / (1024 * 1024)
                            raise ExportError(
                                f"Archive file too large ({file_path.name}); limit is {max_mb:.0f}MB"
                            )

                        total_size += file_size
                        if total_size > total_limit:
                            max_mb = total_limit / (1024 * 1024)
                            raise ExportError(f"Archive size exceeds {max_mb:.0f}MB limit")

                        arcname = file_path.relative_to(work_dir)
                        if arcname.as_posix() == "manifest.json":
                            zf.write(file_path, arcname, compress_type=zipfile.ZIP_STORED)
                        else:
                            zf.write(file_path, arcname)

        # Run in thread pool to avoid blocking
        await asyncio.to_thread(_create_archive)

    def _create_zip_archive(self, work_dir: Path, output_path: Path):
        """Create ZIP archive of the chatbook with compression limits (sync version)."""
        per_file_limit, total_limit = self._get_archive_limits()
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            total_size = 0
            for file_path in work_dir.rglob('*'):
                if file_path.is_file():
                    # Check individual file size
                    file_size = file_path.stat().st_size
                    if file_size > per_file_limit:
                        max_mb = per_file_limit / (1024 * 1024)
                        raise ExportError(
                            f"Archive file too large ({file_path.name}); limit is {max_mb:.0f}MB"
                        )

                    total_size += file_size
                    if total_size > total_limit:
                        max_mb = total_limit / (1024 * 1024)
                        raise ExportError(f"Archive size exceeds {max_mb:.0f}MB limit")

                    arcname = file_path.relative_to(work_dir)
                    if arcname.as_posix() == "manifest.json":
                        zf.write(file_path, arcname, compress_type=zipfile.ZIP_STORED)
                    else:
                        zf.write(file_path, arcname)
