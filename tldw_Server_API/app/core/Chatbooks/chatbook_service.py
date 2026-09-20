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
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from collections.abc import Callable
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

import aiofiles
import aiofiles.os
from loguru import logger

from tldw_Server_API.app.core.config import ACTUAL_PROJECT_ROOT, load_comprehensive_config, settings as core_settings
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.Character_Chat.chat_settings_validation import (
    validate_chat_settings_storage,
)

from ..DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
)
from ..DB_Management.db_path_utils import DatabasePaths
from ..Notes.organization_capture import active_coordinator, capture_note_upsert, stable_note_id
from ..Templating.template_renderer import (
    TemplateContext,
    TemplateEnv,
    options_from_env,
)
from ..Templating.template_renderer import render as render_template
from ..DB_Management.Explainer_DB import ExplainerDatabase
from ..DB_Management.Explainer_Repository import ExplainerRepository
from ..Explainer.chatbook_adapter import (
    EXPLAINER_CHATBOOK_FORMAT,
    restore_explainer_chatbook_payload,
    build_explainer_chatbook_payload,
)

# Legacy job queue shim removed; using in-process task registry
from .chatbook_models import (
    FULL_ACCOUNT_EXPORT_MODE,
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
    coerce_chatbook_export_version,
)
from .chatbook_account_inventory import ACCOUNT_DATA_INVENTORY
from .chatbook_format_v1_1 import (
    build_file_inventory,
    build_preview_report,
    validate_v1_1_before_import,
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
    QuotaExceededError,
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
from ..DB_Management.OpenWebUI_DB import (
    load_openwebui_file_rows_for_ids,
    open_validated_openwebui_db,
    validate_openwebui_file_schema,
)
from .openwebui_folders import (
    build_openwebui_namespace_segments,
    mirror_openwebui_folder_for_conversation,
)
from .openwebui_hydration import (
    MAX_PREVIEW_WARNING_ITEMS,
    OpenWebUIHydrationScope,
    extract_openwebui_hydration_references,
    hydrate_image_reference,
    register_non_image_reference,
    resolve_openwebui_file_path,
    validate_openwebui_data_root,
)
from .quota_manager import QuotaManager, UNLIMITED_QUOTA

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
)
_OPENWEBUI_FOLDER_MIRROR_EXCEPTIONS: tuple[type[BaseException], ...] = (
    *_CHATBOOK_NONCRITICAL_EXCEPTIONS,
    CharactersRAGDBError,
)
_CHATBOOK_SCOPE_COUNT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    *_CHATBOOK_NONCRITICAL_EXCEPTIONS,
    CharactersRAGDBError,
)

_CHATBOOK_TEMPLATE_MODES = {"pass_through", "render_on_export", "render_on_import"}
MAX_OPENWEBUI_HYDRATION_RESPONSE_ITEMS = 1000
ManifestImportPathIndex = dict[tuple[str, str], tuple[str | None, str]]

_ACCOUNT_STATE_SCHEMA_VERSION = "1.0"
_ACCOUNT_PROFILE_ARCHIVE_PATH = "json/account_profile.json"
_ACCOUNT_SETTINGS_ARCHIVE_PATH = "json/account_settings.json"
_ACCOUNT_RESTORE_PAYLOAD_KEY = "_account_restore_payload"

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
        get_unvectorized_chunk_count,
        get_unvectorized_chunks_in_range,
    )
except _CHATBOOK_NONCRITICAL_EXCEPTIONS:  # pragma: no cover
    create_media_database = None  # type: ignore
    get_media_by_id = None  # type: ignore
    get_media_by_uuid = None  # type: ignore
    get_media_transcripts = None  # type: ignore
    get_media_prompts = None  # type: ignore
    get_unvectorized_chunk_count = None  # type: ignore
    get_unvectorized_chunks_in_range = None  # type: ignore

try:
    from ..DB_Management.media_db.legacy_transcripts import upsert_transcript  # type: ignore
except _CHATBOOK_NONCRITICAL_EXCEPTIONS:  # pragma: no cover
    upsert_transcript = None  # type: ignore

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

    _IMPORT_PAYLOAD_FALLBACK_TEMPLATES = {
        ContentType.CONVERSATION: "content/conversations/conversation_{id}.json",
        ContentType.NOTE: "content/notes/note_{id}.md",
        ContentType.CHARACTER: "content/characters/character_{id}.json",
        ContentType.WORLD_BOOK: "content/world_books/world_book_{id}.json",
        ContentType.DICTIONARY: "content/dictionaries/dictionary_{id}.json",
        ContentType.PROMPT: "content/prompts/prompt_{id}.json",
        ContentType.EVALUATION: "content/evaluations/evaluation_{id}.json",
        ContentType.MEDIA: "content/media/media_{id}.json",
        ContentType.EMBEDDING: "content/embeddings/embedding_{id}.json",
        ContentType.GENERATED_DOCUMENT: "content/generated_documents/document_{id}.json",
    }

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
    def _normalize_manifest_archive_path(raw_path: str) -> str | None:
        """Normalize a manifest archive path and reject absolute/traversal paths."""
        if not isinstance(raw_path, str) or not raw_path:
            return None
        normalized = raw_path.replace("\\", "/")
        pure_path = PurePosixPath(normalized)
        if pure_path.is_absolute() or not pure_path.parts:
            return None
        if any(part in {"", ".", ".."} or "\x00" in part for part in pure_path.parts):
            return None
        return pure_path.as_posix()

    @staticmethod
    def _content_type_value(value: Any) -> str | None:
        if value is None:
            return None
        raw_value = getattr(value, "value", value)
        return raw_value if isinstance(raw_value, str) else None

    def _build_manifest_import_path_index(self, manifest: ChatbookManifest) -> ManifestImportPathIndex:
        """Index explicit manifest payload paths by content type and item id."""
        index: ManifestImportPathIndex = {}
        for item in manifest.content_items:
            item_id = str(getattr(item, "id", "") or "")
            type_value = self._content_type_value(getattr(item, "type", None))
            explicit_path = getattr(item, "file_path", None)
            if not item_id or not type_value or not explicit_path:
                continue
            explicit_path_text = str(explicit_path)
            index[(type_value, item_id)] = (
                self._normalize_manifest_archive_path(explicit_path_text),
                explicit_path_text,
            )
        return index

    def _resolve_manifest_import_file(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        content_type: ContentType,
        item_id: str,
        manifest_path_index: ManifestImportPathIndex | None = None,
    ) -> tuple[Path | None, str]:
        """
        Resolve the exact payload path for an import item from the manifest.

        v1.1 validation verifies content item ``file_path`` entries when present;
        import must consume the same path instead of reconstructing fallback names.
        """
        item_id_text = str(item_id)
        fallback_template = self._IMPORT_PAYLOAD_FALLBACK_TEMPLATES.get(content_type)
        rel_path = fallback_template.format(id=item_id_text) if fallback_template else ""

        path_index = (
            manifest_path_index
            if manifest_path_index is not None
            else self._build_manifest_import_path_index(manifest)
        )
        explicit_entry = path_index.get((content_type.value, item_id_text))
        if explicit_entry is not None:
            normalized, display_path = explicit_entry
            if normalized is None:
                return None, display_path
            rel_path = normalized

        if not rel_path:
            return None, item_id_text

        try:
            base_path = extract_dir.resolve()
            candidate = (base_path / rel_path).resolve()
            candidate.relative_to(base_path)
        except (OSError, ValueError):
            return None, rel_path
        return candidate, rel_path

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
    def _coerce_format_version(format_version: ChatbookVersion | str | None) -> ChatbookVersion:
        """Normalize service callers to the canonical ChatbookVersion enum."""
        return coerce_chatbook_export_version(format_version)

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

    @staticmethod
    def _safe_path_component(value: object, *, fallback: str) -> str:
        """Return a filesystem-safe component for generated chatbook files."""
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(value))
        safe = safe.strip("._-")
        return safe or fallback

    def _resolve_export_path(self, name: str, timestamp: str) -> Path:
        """Resolve an export archive path within the user export directory."""
        base = self.export_dir.resolve(strict=False)
        filename = self._build_export_filename(name, timestamp)
        path = (base / filename).resolve(strict=False)
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise SecurityError("Chatbook export path escaped export directory") from exc
        return path

    async def _new_work_dir(self, prefix: str, timestamp: str) -> Path:
        """Create a temporary work directory inside the user chatbook temp directory."""
        base = self.temp_dir.resolve(strict=False)
        await asyncio.to_thread(base.mkdir, parents=True, exist_ok=True, mode=0o700)
        safe_prefix = self._safe_path_component(prefix, fallback="chatbook")
        path = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix=f"{safe_prefix}_{timestamp}_", dir=base))
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise SecurityError("Chatbook work directory escaped temp directory") from exc
        return resolved

    async def _remove_work_dir(self, work_dir: Path) -> None:
        """Remove a chatbook work directory after proving it is under temp_dir."""
        base = self.temp_dir.resolve(strict=False)
        resolved = work_dir.resolve(strict=False)
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise SecurityError("Refusing to remove work directory outside chatbook temp directory") from exc
        await asyncio.to_thread(shutil.rmtree, resolved)

    def __init__(
        self,
        user_id: str | int,
        db: CharactersRAGDB,
        user_id_int: int | None = None,
        user_tier: str = "free",
    ):
        """
        Initialize the chatbook service for a specific user.

        Args:
            user_id: User identifier (string or integer)
            db: User's ChaChaNotes database instance
            user_id_int: Optional integer form of the user id for cross-database access
            user_tier: User quota tier
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
        self.user_tier = user_tier
        self.db = db

        # Track TODOs once per session so we comply with PRD while exposing gaps
        self._todo_messages: set[str] = set()

        # In-process async task registry (best-effort cancellation)
        self._tasks: dict[str, asyncio.Task] = {}
        self._prompts_db: PromptsDatabase | None = None
        self._media_db: Any | None = None
        self._evaluations_db: EvaluationsDatabase | None = None
        self._chroma_manager: ChromaDBManager | None = None
        self._explainer_db: ExplainerDatabase | None = None
        self._explainer_repo: ExplainerRepository | None = None

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

    @classmethod
    def _safe_extracted_content_path(cls, extract_dir: Path, relative_path: str | None) -> Path | None:
        """Resolve a manifest file path inside an extracted chatbook directory."""
        if not relative_path:
            return None
        rel_text = str(relative_path)
        if cls._is_unsafe_archive_path(rel_text):
            return None
        base = extract_dir.resolve()
        target = os.path.normpath(os.path.join(str(base), rel_text))
        try:
            if os.path.commonpath([str(base), target]) != str(base):
                return None
        except ValueError:
            return None
        return Path(target)

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

    def _get_explainer_repo(self) -> ExplainerRepository:
        """Lazily initialize the ownership-scoped Explainer repository."""
        if self._explainer_repo is not None:
            return self._explainer_repo
        user_id_value = self.user_id_int if self.user_id_int is not None else self.user_id
        db_path = DatabasePaths.get_explainer_db_path(user_id_value)
        self._explainer_db = ExplainerDatabase(db_path=db_path, client_id=self.user_id)
        self._explainer_repo = ExplainerRepository(self._explainer_db)
        return self._explainer_repo

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
    def _serialize_job_timestamp(value: datetime | None) -> str | None:
        """Serialize a job timestamp with explicit UTC semantics."""
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat(sep=" ")

    @staticmethod
    def _job_timestamp_for_api(value: datetime | None) -> datetime | None:
        """Return an aware UTC timestamp for API serialization."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @classmethod
    def _normalize_job_timestamps_for_api(cls, job: ExportJob | ImportJob) -> None:
        for field_name in ("created_at", "started_at", "completed_at", "expires_at"):
            if hasattr(job, field_name):
                setattr(job, field_name, cls._job_timestamp_for_api(getattr(job, field_name)))

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

    def _normalize_media_file_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Normalize MediaFiles metadata without leaking server-only path details."""
        return {
            "id": row.get("id"),
            "uuid": row.get("uuid"),
            "file_type": row.get("file_type"),
            "original_filename": row.get("original_filename"),
            "file_size": row.get("file_size"),
            "mime_type": row.get("mime_type"),
            "checksum": row.get("checksum"),
            "created_at": self._normalize_datetime(row.get("created_at")),
            "last_modified": self._normalize_datetime(row.get("last_modified")),
        }

    def _resolve_owned_media_artifact_path(self, storage_path: Any) -> Path | None:
        """Resolve stored media bytes only when they live under the account-owned user root."""
        raw = str(storage_path or "").strip()
        if not raw:
            return None
        try:
            user_root = DatabasePaths.resolve_user_base_directory(self.user_id_int or self.user_id).resolve()
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
            return None

        candidates = [Path(raw)] if Path(raw).is_absolute() else [user_root / raw]
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                resolved.relative_to(user_root)
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                continue
            if resolved.is_file():
                return resolved
        return None

    def _copy_media_artifacts(
        self,
        media_db: Any,
        media_id: int,
        media_dir: Path,
        media_id_text: str,
    ) -> tuple[list[dict[str, Any]], int, int]:
        list_files = getattr(media_db, "get_media_files", None)
        if not callable(list_files):
            return [], 0, 0
        try:
            rows = list_files(media_id)
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
            return [], 0, 0

        files_dir = media_dir / "files" / f"media_{media_id_text}"
        exported: list[dict[str, Any]] = []
        bundled_count = 0
        pointer_count = 0
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            item = self._normalize_media_file_row(row)
            source_path = self._resolve_owned_media_artifact_path(row.get("storage_path"))
            if source_path is None:
                item["bundled"] = False
                item["pointer_only"] = True
                pointer_count += 1
                exported.append(item)
                continue

            files_dir.mkdir(parents=True, exist_ok=True)
            suffix = source_path.suffix or ".bin"
            file_id = str(row.get("id") or row.get("uuid") or len(exported))
            dest_name = f"file_{file_id}{suffix}"
            dest = files_dir / dest_name
            try:
                shutil.copy2(source_path, dest)
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                item["bundled"] = False
                item["pointer_only"] = True
                pointer_count += 1
            else:
                item["bundled"] = True
                item["pointer_only"] = False
                item["archive_path"] = f"content/media/files/media_{media_id_text}/{dest_name}"
                bundled_count += 1
            exported.append(item)
        return exported, bundled_count, pointer_count

    @staticmethod
    def _coerce_inventory_count(value: Any) -> int:
        """Return a non-negative inventory count for import summaries."""
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def _inspect_manifest_inventory_for_import(
        self,
        manifest: ChatbookManifest,
    ) -> tuple[dict[str, Any] | None, dict[str, int], list[str], list[str]]:
        """Build redacted inventory import context and fail-closed coverage errors."""
        summary = manifest.account_inventory_summary or {}
        redacted_summary = dict(summary) if isinstance(summary, dict) and summary else None
        counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
        if not isinstance(counts, dict):
            counts = {}

        known_categories = {entry.category for entry in ACCOUNT_DATA_INVENTORY}
        inventory_definitions = {entry.category: entry for entry in ACCOUNT_DATA_INVENTORY}
        canonical_warnings = {
            entry.category: entry.warning
            for entry in ACCOUNT_DATA_INVENTORY
            if entry.warning
        }
        skipped_non_restorable: dict[str, int] = {}
        warnings: list[str] = []
        errors: list[str] = []
        serialized_content_categories = {
            "conversations",
            "notes",
            "characters",
            "world_books",
            "dictionaries",
            "prompts",
            "evaluations",
            "generated_documents",
            "explainer_sessions",
            "media_records",
            "media_transcripts",
            "media_chunks",
            "media_stored_artifacts",
            "embeddings",
        }
        account_metadata_without_payload = {"tags_categories_relationships"}
        serialized_account_categories = {"account_profile", "account_settings"}

        for entry in manifest.account_inventory or []:
            if not isinstance(entry, dict):
                continue
            category = str(entry.get("category") or "").strip()
            if not category:
                continue
            restore_status = str(entry.get("restore_status") or "").strip()
            definition = inventory_definitions.get(category)
            manifest_count_key = definition.manifest_count_key if definition else category
            count = self._coerce_inventory_count(
                counts.get(manifest_count_key, counts.get(category))
            )
            if restore_status == "restorable" and category not in known_categories:
                errors.append(
                    f"Archive inventory category '{category}' is marked restorable but this importer has no handler."
                )
                continue
            if restore_status in {"pointer_only", "non_restorable"}:
                if count:
                    skipped_non_restorable[category] = count
                warning = canonical_warnings.get(category)
                if warning:
                    warnings.append(str(warning))
                elif category not in known_categories:
                    warnings.append(
                        f"Archive inventory category '{category}' is {restore_status} and was not restored."
                    )
                continue
            if (
                restore_status == "restorable"
                and count
                and category in account_metadata_without_payload
            ):
                skipped_non_restorable[category] = count
                warnings.append(
                    f"Archive inventory category '{category}' has no serialized restore payload; "
                    "review account settings after import."
                )
            elif (
                restore_status == "restorable"
                and count
                and category not in serialized_content_categories
                and category not in serialized_account_categories
            ):
                warnings.append(
                    f"Archive inventory category '{category}' is known but has no direct content item payload."
                )

        return redacted_summary, skipped_non_restorable, warnings, errors

    def _load_account_restore_payloads(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """Load and validate versioned account payloads declared by inventory counts."""
        summary = manifest.account_inventory_summary or {}
        counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
        counts = counts if isinstance(counts, dict) else {}
        manifest_inventory = {
            str(row.get("category") or ""): row
            for row in manifest.account_inventory or []
            if isinstance(row, dict)
        }
        definitions = {entry.category: entry for entry in ACCOUNT_DATA_INVENTORY}
        inventory_paths = {
            normalized_path
            for entry in manifest.file_inventory or []
            if isinstance(entry, dict)
            and (
                normalized_path := self._normalize_manifest_archive_path(
                    str(entry.get("path") or "")
                )
            )
        }
        payloads: dict[str, dict[str, Any]] = {}
        errors: list[str] = []

        for category, canonical_path in (
            ("account_profile", _ACCOUNT_PROFILE_ARCHIVE_PATH),
            ("account_settings", _ACCOUNT_SETTINGS_ARCHIVE_PATH),
        ):
            definition = definitions[category]
            count = self._coerce_inventory_count(
                counts.get(definition.manifest_count_key, counts.get(category))
            )
            inventory_row = manifest_inventory.get(category, {})
            declared_path = str(inventory_row.get("export_representation") or canonical_path)
            normalized_path = self._normalize_manifest_archive_path(declared_path)
            if normalized_path != canonical_path:
                errors.append(
                    f"Archive inventory category '{category}' has an invalid account payload path."
                )
                continue
            payload_path = extract_dir / canonical_path
            if count == 0:
                if payload_path.exists():
                    errors.append(
                        f"Archive account payload '{canonical_path}' is present but its inventory count is zero."
                    )
                continue
            if count != 1:
                errors.append(
                    f"Archive inventory category '{category}' must declare exactly one account payload."
                )
                continue
            if not payload_path.is_file():
                errors.append(
                    f"Archive inventory category '{category}' is missing its serialized restore payload."
                )
                continue
            if (
                manifest.version == ChatbookVersion.V1_1
                and canonical_path not in inventory_paths
            ):
                errors.append(
                    f"Archive inventory category '{category}' is missing verified file inventory entry."
                )
                continue
            try:
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                errors.append(f"Archive inventory category '{category}' has an unreadable payload.")
                continue
            if not isinstance(payload, dict):
                errors.append(f"Archive inventory category '{category}' payload must be an object.")
                continue
            if payload.get("schema_version") != _ACCOUNT_STATE_SCHEMA_VERSION:
                errors.append(
                    f"Archive inventory category '{category}' uses an unsupported account payload version."
                )
                continue
            if payload.get("category") != category:
                errors.append(f"Archive inventory category '{category}' payload type does not match.")
                continue
            if category == "account_profile" and not isinstance(payload.get("profile"), dict):
                errors.append("Archive account profile payload is missing its profile object.")
                continue
            if category == "account_settings" and not isinstance(payload.get("overrides"), dict):
                errors.append("Archive account settings payload is missing its overrides object.")
                continue
            payloads[category] = payload

        return payloads, errors

    def _safe_import_artifact_destination(
        self,
        *,
        media_id: int,
        original_filename: Any,
        fallback_name: str,
    ) -> tuple[Path, str] | None:
        """Return an account-owned destination for a bundled media artifact."""
        try:
            user_root = DatabasePaths.resolve_user_base_directory(self.user_id_int or self.user_id).resolve()
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
            return None
        raw_name = str(original_filename or fallback_name or "artifact.bin")
        safe_name = Path(raw_name.replace("\\", "/")).name or "artifact.bin"
        if self._is_unsafe_archive_path(safe_name):
            safe_name = "artifact.bin"
        dest_dir = user_root / "imported_media" / f"media_{media_id}"
        dest_name = f"{uuid4().hex}_{safe_name}"
        dest = (dest_dir / dest_name).resolve()
        try:
            dest.relative_to(user_root)
        except ValueError:
            return None
        return dest, dest.relative_to(user_root).as_posix()

    def _restore_media_artifacts(
        self,
        *,
        media_db: Any,
        extract_dir: Path,
        media_id: int,
        stored_artifacts: list[dict[str, Any]],
        status: ImportJob,
    ) -> tuple[int, int]:
        """Restore bundled media artifact bytes and count pointer-only metadata."""
        restored = 0
        pointer_only = 0
        insert_media_file = getattr(media_db, "insert_media_file", None)
        if not callable(insert_media_file):
            status.warnings.append("Media artifact rows skipped: media DB does not support MediaFiles restore.")
            return restored, len(stored_artifacts or [])

        for index, artifact in enumerate(stored_artifacts or []):
            if not isinstance(artifact, dict):
                continue
            archive_path = artifact.get("archive_path")
            if not artifact.get("bundled") or not archive_path:
                pointer_only += 1
                continue
            source = self._safe_extracted_content_path(extract_dir, str(archive_path))
            if source is None or not source.exists() or not source.is_file():
                pointer_only += 1
                status.warnings.append(f"Stored media artifact missing from archive: {archive_path}")
                continue
            destination = self._safe_import_artifact_destination(
                media_id=media_id,
                original_filename=artifact.get("original_filename"),
                fallback_name=Path(str(archive_path)).name or f"artifact_{index}.bin",
            )
            if destination is None:
                pointer_only += 1
                status.warnings.append("Stored media artifact skipped: could not resolve account-owned destination.")
                continue
            dest_path, storage_path = destination
            try:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest_path)
                insert_media_file(
                    media_id=int(media_id),
                    file_type=str(artifact.get("file_type") or "artifact"),
                    storage_path=storage_path,
                    original_filename=artifact.get("original_filename"),
                    file_size=int(artifact.get("file_size") or source.stat().st_size),
                    mime_type=artifact.get("mime_type"),
                    checksum=artifact.get("checksum"),
                )
                restored += 1
            except Exception as exc:
                pointer_only += 1
                status.warnings.append(f"Stored media artifact restore failed: {str(exc)}")
        return restored, pointer_only

    @staticmethod
    def _media_chunks_for_restore(payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert exported media chunks into the media DB writer shape."""
        chunks: list[dict[str, Any]] = []
        for chunk in payload.get("chunks") or []:
            if not isinstance(chunk, dict):
                continue
            text = chunk.get("text")
            if text is None:
                text = chunk.get("chunk_text")
            if text is None:
                continue
            chunks.append(
                {
                    "text": text,
                    "start_char": chunk.get("start_char"),
                    "end_char": chunk.get("end_char"),
                    "chunk_type": chunk.get("chunk_type"),
                    "metadata": chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {},
                }
            )
        return chunks

    def _restore_media_transcripts(
        self,
        *,
        media_db: Any,
        media_id: int,
        payload: dict[str, Any],
        status: ImportJob,
    ) -> None:
        """Restore exported transcript rows for a media item when supported."""
        if upsert_transcript is None:
            if payload.get("transcripts"):
                status.warnings.append("Media transcripts skipped: transcript restore API unavailable.")
            return
        for transcript in payload.get("transcripts") or []:
            if not isinstance(transcript, dict):
                continue
            text = (
                transcript.get("transcription")
                or transcript.get("text")
                or transcript.get("content")
            )
            if not text:
                continue
            try:
                upsert_transcript(
                    media_db,
                    int(media_id),
                    str(text),
                    str(transcript.get("whisper_model") or transcript.get("model") or "imported"),
                    created_at=transcript.get("created_at"),
                    idempotency_key=f"chatbook:{payload.get('uuid') or payload.get('id')}:{transcript.get('id') or uuid4().hex}",
                    set_as_latest=True,
                )
            except Exception as exc:
                status.warnings.append(f"Media transcript restore failed: {str(exc)}")

    def _restore_media_vector_embedding(
        self,
        media_db: Any,
        media_id: int,
        vector_payload: dict[str, Any],
    ) -> None:
        """Attach an exported media vector blob back to a restored media row."""
        if vector_payload.get("encoding") != "base64" or not vector_payload.get("vector"):
            return
        vector_bytes = base64.b64decode(str(vector_payload["vector"]))
        with media_db.transaction() as conn:
            if not hasattr(media_db, "_fetchone_with_connection"):
                media_db._execute_with_connection(
                    conn,
                    "UPDATE Media SET vector_embedding = ? WHERE id = ?",
                    (vector_bytes, int(media_id)),
                )
                return
            media_row = media_db._fetchone_with_connection(
                conn,
                "SELECT uuid, version FROM Media WHERE id = ? AND deleted = 0",
                (int(media_id),),
            )
            if not media_row:
                raise ValueError("Restored media row was not found for vector attachment")
            current_version = int(media_row.get("version") or 1)
            next_version = current_version + 1
            now = media_db._get_current_utc_timestamp_str()
            client_id = media_db.client_id
            cursor = media_db._execute_with_connection(
                conn,
                """
                UPDATE Media
                   SET vector_embedding = ?, last_modified = ?, version = ?, client_id = ?
                 WHERE id = ? AND version = ? AND deleted = 0
                """,
                (vector_bytes, now, next_version, client_id, int(media_id), current_version),
            )
            if getattr(cursor, "rowcount", 0) != 1:
                raise ValueError("Restored media row changed during vector attachment")
            media_db._log_sync_event(
                conn,
                "Media",
                str(media_row["uuid"]),
                "update",
                next_version,
                {
                    "id": int(media_id),
                    "uuid": str(media_row["uuid"]),
                    "version": next_version,
                    "last_modified": now,
                    "client_id": client_id,
                    "vector_embedding_restored": True,
                },
            )

    def _filter_chroma_ids_for_conflict_resolution(
        self,
        *,
        chroma: Any,
        collection_name: str,
        ids: list[str],
        conflict_resolution: ConflictResolution,
        status: ImportJob,
    ) -> set[str]:
        """Return Chroma ids safe to upsert for the selected conflict mode."""
        if conflict_resolution != ConflictResolution.SKIP or not ids:
            return set(ids)
        try:
            collection = chroma.get_collection(collection_name)
        except KeyError:
            return set(ids)
        except Exception as exc:
            status.warnings.append(
                f"Embedding collection conflict check failed for '{collection_name}': {str(exc)}"
            )
            return set()

        try:
            result = collection.get(ids=ids)
        except Exception as exc:
            status.warnings.append(
                f"Embedding collection conflict check failed for '{collection_name}': {str(exc)}"
            )
            return set()
        existing_ids = {
            str(item_id)
            for item_id in (result.get("ids") or [])
            if item_id is not None
        }
        if existing_ids:
            status.warnings.append(
                f"Skipped {len(existing_ids)} existing embedding id(s) in collection '{collection_name}'."
            )
        return {item_id for item_id in ids if item_id not in existing_ids}

    @staticmethod
    def _renamed_import_title(base_name: str) -> str:
        """Return a low-collision title for content types without local name lookup."""
        return f"{base_name} (Imported {uuid4().hex[:8]})"

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
        def _job_table_has_column(table_name: str, column_name: str) -> bool:
            if table_name not in {"export_jobs", "import_jobs"}:
                raise ValueError(f"Unsupported Chatbook job table: {table_name}")
            if self._uses_postgres_backend():
                cursor = self.db.execute_query(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = ? AND column_name = ?
                    """,
                    (table_name, column_name),
                )
            else:
                cursor = self.db.execute_query(
                    f"PRAGMA table_info({table_name})"  # nosec B608 - table name is allowlisted above.
                )
            for row in self._fetch_results(cursor) or []:
                if isinstance(row, dict):
                    candidate = row.get("name") or row.get("column_name")
                else:
                    candidate = row[1] if len(row) > 1 else None
                if candidate == column_name:
                    return True
            return False

        def _ensure_job_table_column(table_name: str, column_name: str, definition: str) -> None:
            if not _job_table_has_column(table_name, column_name):
                self.db.execute_query(f"ALTER TABLE {table_name} ADD COLUMN {definition}")  # nosec B608

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
                    expires_at TIMESTAMP,
                    metadata TEXT
                )
            """)
            _ensure_job_table_column("export_jobs", "metadata", "metadata TEXT")

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
                    warnings TEXT,    -- JSON array
                    metadata TEXT     -- JSON object
                )
            """)
            _ensure_job_table_column("import_jobs", "metadata", "metadata TEXT")
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
                                'explainer_sessions': stats.get(
                                    'total_explainer_sessions',
                                    manifest_data.get('total_explainer_sessions', 0),
                                ),
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
        content_selections: dict[ContentType, list[str]] | None,
        author: str | None = None,
        include_media: bool = False,
        media_quality: str = "compressed",
        include_embeddings: bool = False,
        include_generated_content: bool = True,
        tags: list[str] | None = None,
        categories: list[str] | None = None,
        format_version: ChatbookVersion = ChatbookVersion.V1,
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
            format_version: Chatbook manifest format version to produce
            async_mode: Run as background job

        Returns:
            Tuple of (success, message, job_id or file_path)
        """
        format_version = self._coerce_format_version(format_version)
        selection_mode = (
            FULL_ACCOUNT_EXPORT_MODE
            if content_selections is None or content_selections == {}
            else "allowlist"
        )
        if selection_mode == "allowlist" and sum(len(ids or []) for ids in content_selections.values()) == 0:
            return False, "Export allowlist contains no exportable items.", None

        if not async_mode:
            self._check_chatbook_job_admission_with_lock("export")

        if async_mode:
            # Create job and run asynchronously
            job_id = str(uuid4())
            job = ExportJob(
                job_id=job_id,
                user_id=self.user_id,
                status=ExportStatus.PENDING,
                chatbook_name=name
            )

            # Store job in database after atomically reserving Chatbooks quota.
            self._save_export_job_with_quota(job)

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
                    "selection_mode": selection_mode,
                    "content_selections": (
                        None
                        if selection_mode == FULL_ACCOUNT_EXPORT_MODE
                        else {k.value if hasattr(k, 'value') else str(k): v for k, v in content_selections.items()}
                    ),
                    "author": author,
                    "include_media": include_media,
                    "media_quality": media_quality,
                    "include_embeddings": include_embeddings,
                    "include_generated_content": include_generated_content,
                    "tags": tags or [],
                    "categories": categories or [],
                    "format_version": format_version.value if hasattr(format_version, "value") else str(format_version),
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

            return True, f"Export job started: {job_id}", job_id
        else:
            # Run synchronously (wrapped in async)
            return await self._create_chatbook_sync_wrapper(
                name, description, content_selections,
                author, include_media, media_quality, include_embeddings,
                include_generated_content, tags, categories,
                format_version=format_version,
                selection_mode=selection_mode,
            )

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
            work_dir = await self._new_work_dir("continue", timestamp)

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

                    safe_eval_id = self._safe_path_component(eval_id, fallback="evaluation")
                    eval_file = eval_dir / f"evaluation_{safe_eval_id}_cont.json"
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
            output_path = self._resolve_export_path(cont_name, timestamp)
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
                    await self._remove_work_dir(work_dir)
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                    pass

    async def _write_full_account_state_payloads(
        self,
        work_dir: Path,
        manifest: ChatbookManifest,
    ) -> None:
        """Serialize portable AuthNZ profile state and user-owned settings overrides."""
        if self.user_id_int is None:
            raise ExportError("Full-account profile export requires a numeric user id")

        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
        from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
        from tldw_Server_API.app.core.UserProfiles.overrides_repo import UserProfileOverridesRepo
        from tldw_Server_API.app.core.UserProfiles.user_profile_catalog import (
            load_user_profile_catalog,
        )

        db_pool = await get_db_pool()
        user = await AuthnzUsersRepo(db_pool=db_pool).get_user_by_id(self.user_id_int)
        if user is None:
            raise ExportError("Full-account profile export could not resolve the account record")

        catalog = load_user_profile_catalog()
        catalog_entries = {str(entry.key): entry for entry in catalog.entries}
        profile: dict[str, Any] = {}
        email = str(user.get("email") or "").strip()
        if email:
            profile["identity.email"] = email

        overrides_repo = UserProfileOverridesRepo(db_pool)
        await overrides_repo.ensure_tables()
        override_rows = await overrides_repo.list_overrides_for_user(self.user_id_int)
        overrides: dict[str, Any] = {}
        unrecognized_overrides: list[dict[str, Any]] = []
        for row in override_rows:
            key = str(row.get("key") or "").strip()
            if not key:
                continue
            entry = catalog_entries.get(key)
            if entry is None or "user" not in set(entry.editable_by or []):
                unrecognized_overrides.append({"key": key, "value": row.get("value")})
                continue
            overrides[key] = row.get("value")

        policy = {
            "destination_owned_fields": [
                "identity.id",
                "identity.uuid",
                "identity.username",
                "identity.role",
                "identity.is_active",
                "identity.is_verified",
            ],
            "excluded_secret_categories": [
                "password_hashes",
                "sessions",
                "api_keys",
                "provider_credentials",
                "deployment_secrets",
            ],
            "exclusion_reason": "Authentication, authorization, and deployment-bound secrets require destination reconfiguration.",
        }
        profile_payload = {
            "schema_version": _ACCOUNT_STATE_SCHEMA_VERSION,
            "category": "account_profile",
            "profile": profile,
            "policy": policy,
        }
        settings_payload = {
            "schema_version": _ACCOUNT_STATE_SCHEMA_VERSION,
            "category": "account_settings",
            "catalog_version": catalog.version,
            "overrides": overrides,
            "unrecognized_overrides": unrecognized_overrides,
            "policy": {
                "restore_contract": "catalog_controlled_user_overrides",
                "unrecognized_override_behavior": "preserved_in_archive_not_restored",
            },
        }

        json_dir = work_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        for path, payload in (
            (json_dir / "account_profile.json", profile_payload),
            (json_dir / "account_settings.json", settings_payload),
        ):
            async with aiofiles.open(path, "w", encoding="utf-8") as handle:
                await handle.write(json.dumps(payload, indent=2, ensure_ascii=False))
            with contextlib.suppress(OSError):
                path.chmod(0o600)

        manifest.metadata["account_state_counts"] = {
            "account_profiles": 1,
            "account_settings": 1,
        }
        manifest.metadata["account_state_policy"] = {
            "schema_version": _ACCOUNT_STATE_SCHEMA_VERSION,
            "destination_owned_identity": True,
            "authentication_secrets_restored": False,
            "unrecognized_setting_count": len(unrecognized_overrides),
        }
        if unrecognized_overrides:
            manifest.metadata.setdefault("pointer_only_warnings", []).append(
                f"{len(unrecognized_overrides)} unrecognized account setting override(s) were preserved but cannot be restored by this version."
            )

    def _build_account_inventory_summary(
        self,
        content: ChatbookContent,
        archive_size_bytes: int = 0,
        post_write_verification: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        media_source_pointer_count = sum(
            1 for item in content.media.values() if item.get("url")
        )
        media_artifact_pointer_count = sum(
            1
            for item in content.media.values()
            for artifact in item.get("stored_artifacts") or []
            if artifact.get("pointer_only")
        )
        account_state_counts = (
            metadata.get("account_state_counts", {})
            if isinstance(metadata, dict)
            else {}
        )
        counts = {
            "account_profiles": self._coerce_inventory_count(
                account_state_counts.get("account_profiles")
            ),
            "account_settings": self._coerce_inventory_count(
                account_state_counts.get("account_settings")
            ),
            "conversations": len(content.conversations),
            "notes": len(content.notes),
            "characters": len(content.characters),
            "world_books": len(content.world_books),
            "dictionaries": len(content.dictionaries),
            "prompts": len(content.prompts),
            "evaluations": len(content.evaluations),
            "generated_documents": len(content.generated_documents),
            "explainer_sessions": len(content.explainer_sessions),
            "media_records": len(content.media),
            "media_transcripts": sum(len(item.get("transcripts") or []) for item in content.media.values()),
            "media_chunks": sum(len(item.get("chunks") or []) for item in content.media.values()),
            "media_stored_artifacts": sum(
                1
                for item in content.media.values()
                for artifact in item.get("stored_artifacts") or []
                if artifact.get("bundled")
            ),
            "media_pointers": media_source_pointer_count + media_artifact_pointer_count,
            "embeddings": len(content.embeddings),
            "tags_categories_relationships": self._scope_db_count_for_category("tags_categories_relationships"),
            "sensitive_user_values": 0,
        }
        warnings = [row.warning for row in ACCOUNT_DATA_INVENTORY if row.warning]
        if metadata:
            warnings.extend(
                str(warning)
                for warning in metadata.get("pointer_only_warnings", []) or []
                if warning
            )
        return {
            "counts": counts,
            "pointer_only_count": sum(
                counts.get(row.manifest_count_key, 0)
                for row in ACCOUNT_DATA_INVENTORY
                if row.restore_status == "pointer_only"
            ),
            "sensitive_category_count": sum(
                1 for row in ACCOUNT_DATA_INVENTORY if row.sensitivity in {"sensitive", "secret"}
            ),
            "warning_count": len(warnings),
            "warnings": warnings,
            "archive_size_bytes": archive_size_bytes,
            "post_write_verification": post_write_verification,
        }

    @staticmethod
    def _verify_export_archive(output_path: Path) -> tuple[bool, str | None]:
        try:
            archive_size = output_path.stat().st_size
            with zipfile.ZipFile(output_path, "r") as zf:
                names = set(zf.namelist())
                if "manifest.json" not in names:
                    return False, "manifest.json missing from archive"
                manifest = json.loads(zf.read("manifest.json"))
                for item in manifest.get("content_items") or []:
                    file_path = item.get("file_path")
                    if file_path and file_path not in names:
                        return False, f"manifest file_path missing from archive: {file_path}"
                statistics = manifest.get("statistics") or {}
                if int(statistics.get("total_size_bytes") or 0) != archive_size:
                    return False, "manifest total_size_bytes does not match ZIP size"
                summary = manifest.get("account_inventory_summary") or {}
                if summary and int(summary.get("archive_size_bytes") or 0) != archive_size:
                    return False, "manifest archive_size_bytes does not match ZIP size"
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
            return False, str(exc)
        return True, None

    @staticmethod
    def build_export_job_metadata(output_path: str | Path | None) -> dict[str, Any]:
        """Read redacted completed-export metadata from a Chatbook archive."""
        if not output_path:
            return {}
        try:
            with zipfile.ZipFile(output_path, "r") as zf:
                manifest = json.loads(zf.read("manifest.json"))
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
            return {}
        summary = manifest.get("account_inventory_summary") or {}
        counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
        total_items = sum(
            max(0, int(count or 0))
            for count in counts.values()
            if isinstance(count, (int, float)) and not isinstance(count, bool)
        )
        return {
            "account_inventory_summary": summary,
            "archive_size_bytes": summary.get("archive_size_bytes")
            or (manifest.get("statistics") or {}).get("total_size_bytes"),
            "post_write_verification": bool(summary.get("post_write_verification")),
            "pointer_only_count": int(summary.get("pointer_only_count") or 0),
            "warning_count": int(summary.get("warning_count") or 0),
            "sensitive_category_count": int(summary.get("sensitive_category_count") or 0),
            "total_items": total_items,
        }

    async def _stabilize_archive_size(
        self,
        work_dir: Path,
        output_path: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent,
        write_manifest: Callable[[], Any],
    ) -> None:
        for _ in range(10):
            archive_size = output_path.stat().st_size
            if manifest.total_size_bytes == archive_size:
                break
            manifest.total_size_bytes = archive_size
            manifest.account_inventory_summary = self._build_account_inventory_summary(
                content,
                archive_size_bytes=archive_size,
                post_write_verification=bool(manifest.account_inventory_summary.get("post_write_verification")),
                metadata=manifest.metadata,
            )
            await write_manifest()
            await self._create_zip_archive_async(work_dir, output_path)

    async def _create_chatbook_sync_wrapper(
        self,
        name: str,
        description: str,
        content_selections: dict[ContentType, list[str]] | None,
        author: str | None = None,
        include_media: bool = False,
        media_quality: str = "compressed",
        include_embeddings: bool = False,
        include_generated_content: bool = True,
        tags: list[str] | None = None,
        categories: list[str] | None = None,
        format_version: ChatbookVersion = ChatbookVersion.V1,
        selection_mode: str = "allowlist",
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
            work_dir = await self._new_work_dir("export", timestamp)

            manifest_metadata = self._default_chatbook_template_metadata()
            manifest_metadata["selection_mode"] = selection_mode

            # Initialize manifest
            manifest = ChatbookManifest(
                version=self._coerce_format_version(format_version),
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
                metadata=manifest_metadata,
            )
            manifest.binary_limits = self._get_binary_limits_bytes()
            if selection_mode == FULL_ACCOUNT_EXPORT_MODE:
                content_selections = self._expand_full_account_content_selections()
                include_media = True
                include_embeddings = True
                include_generated_content = True
                manifest.include_media = True
                manifest.include_embeddings = True
                manifest.include_generated_content = True
                await self._write_full_account_state_payloads(work_dir, manifest)
            if manifest.version == ChatbookVersion.V1_1:
                manifest.features_used = [
                    "content_envelopes",
                    "file_inventory",
                    "integrity_metadata",
                    "representations",
                    "lossiness_metadata",
                ]
                manifest.producer = {"name": "tldw_server"}
                manifest.source_instance = {}
                manifest.compatibility = {
                    "min_reader_version": "1.1.0",
                    "recommended_reader_version": "1.1.0",
                    "unsupported_feature_behavior": "warn_lossy_import",
                    "v1_compatibility": {
                        "fallback": (
                            "Readers that only support v1.0 may use the core manifest fields "
                            "and ignore v1.1 metadata, with possible loss of representation "
                            "and integrity details."
                        )
                    },
                }

            # Collect content
            content = ChatbookContent()
            content_selections = content_selections or {}

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

            if ContentType.EXPLAINER_SESSION in content_selections:
                self._collect_explainer_sessions(
                    content_selections[ContentType.EXPLAINER_SESSION],
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
            manifest.total_explainer_sessions = len(content.explainer_sessions)
            manifest.account_inventory = [row.to_summary() for row in ACCOUNT_DATA_INVENTORY]
            manifest.account_inventory_summary = self._build_account_inventory_summary(
                content,
                metadata=manifest.metadata,
            )

            # Write manifest asynchronously
            manifest_path = work_dir / "manifest.json"
            async def _write_manifest() -> None:
                """Write the current manifest to disk as formatted JSON."""
                async with aiofiles.open(manifest_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
            await _write_manifest()

            # Create README asynchronously
            await self._create_readme_async(work_dir, manifest)
            if manifest.version == ChatbookVersion.V1_1:
                manifest.file_inventory = await asyncio.to_thread(build_file_inventory, work_dir)
                await _write_manifest()

            # Create archive in secure export directory
            output_path = self._resolve_export_path(name, timestamp)
            await self._create_zip_archive_async(work_dir, output_path)

            # Update manifest with final archive size; re-zip if manifest changes size.
            await self._stabilize_archive_size(work_dir, output_path, manifest, content, _write_manifest)
            verified, verification_error = self._verify_export_archive(output_path)
            if not verified:
                raise ExportError(f"Chatbook archive verification failed: {verification_error}")
            manifest.account_inventory_summary = self._build_account_inventory_summary(
                content,
                archive_size_bytes=manifest.total_size_bytes,
                post_write_verification=True,
                metadata=manifest.metadata,
            )
            await _write_manifest()
            await self._create_zip_archive_async(work_dir, output_path)
            await self._stabilize_archive_size(work_dir, output_path, manifest, content, _write_manifest)
            verified, verification_error = self._verify_export_archive(output_path)
            if not verified:
                raise ExportError(f"Chatbook archive verification failed: {verification_error}")

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
                    await self._remove_work_dir(work_dir)
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as cleanup_err:
                    logger.warning(f"Failed to remove work directory {work_dir}: {cleanup_err}")

    async def _create_chatbook_job_async(
        self,
        job_id: str,
        name: str,
        description: str,
        content_selections: dict[ContentType, list[str]] | None,
        author: str | None,
        include_media: bool,
        media_quality: str,
        include_embeddings: bool,
        include_generated_content: bool,
        tags: list[str],
        categories: list[str],
        format_version: ChatbookVersion = ChatbookVersion.V1,
        selection_mode: str = "allowlist",
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

            # Create chatbook using the sync wrapper (could be made truly async)
            success, message, file_path = await self._create_chatbook_sync_wrapper(
                name, description, content_selections,
                author, include_media, media_quality, include_embeddings,
                include_generated_content, tags, categories,
                format_version=format_version,
                selection_mode=selection_mode,
            )

            if success:
                # Update job with success; respect cancellation
                latest = self._get_export_job(job.job_id)
                if latest and latest.status == ExportStatus.CANCELLED:
                    return
                job.status = ExportStatus.COMPLETED
                now_utc = datetime.now(timezone.utc)
                job.completed_at = now_utc
                job.output_path = file_path
                job.file_size_bytes = Path(file_path).stat().st_size if file_path else None
                terminal_metadata = self.build_export_job_metadata(file_path)
                job.total_items = int(terminal_metadata.get("total_items") or 0)
                job.processed_items = job.total_items
                job.progress_percentage = 100
                job.metadata = {
                    **(job.metadata if isinstance(job.metadata, dict) else {}),
                    **terminal_metadata,
                }
                # Build (optionally signed) download URL and expiry
                job.expires_at = self._get_export_expiry(now_utc)
                download_expires_at = self._get_download_expiry(now_utc, job.expires_at)
                job.download_url = self._build_download_url(job.job_id, download_expires_at)
            else:
                # Update job with failure
                job.status = ExportStatus.FAILED
                job.completed_at = datetime.now(timezone.utc)
                job.error_message = message
            self._save_export_job(job)

        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            # Update job with error
            job.status = ExportStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error_message = str(e)
            self._save_export_job(job)

    async def import_chatbook(
        self,
        file_path: str,
        content_selections: dict[ContentType, list[str]] | None = None,
        conflict_resolution: ConflictResolution | str | None = None,
        conflict_strategy: str | None = None,  # Alias for conflict_resolution (for test compatibility)
        prefix_imported: bool = False,
        import_media: bool | None = None,
        import_embeddings: bool | None = None,
        async_mode: bool = False,
        request_id: str | None = None,
        source_format: str = "chatbook",
        selected_openwebui_user_id: str | None = None,
        source_filename: str | None = None,
    ) -> tuple[bool, str, str | dict[str, Any] | None]:
        """
        Import a chatbook.

        Args:
            file_path: Path to chatbook file
            source_filename: User-facing upload filename; server paths are removed
            content_selections: Specific content to import
            conflict_resolution: How to handle conflicts
            prefix_imported: Add prefix to imported items
            import_media: Import archive media files; defaults to true for Chatbook archives
            import_embeddings: Import archive embeddings; defaults to true for Chatbook archives
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

        source_format_value = getattr(source_format, "value", str(source_format or "chatbook")).strip().lower()
        if source_format_value not in {"chatbook", "openwebui_json", "openwebui_db"}:
            return False, f"Unsupported import source format: {source_format}", None

        effective_import_media = (
            bool(import_media)
            if import_media is not None
            else source_format_value == "chatbook"
        )
        effective_import_embeddings = (
            bool(import_embeddings)
            if import_embeddings is not None
            else source_format_value == "chatbook"
        )

        if source_format_value in {"openwebui_json", "openwebui_db"}:
            if content_selections:
                return False, "OpenWebUI imports do not support content selections; all valid chats are imported.", None
            if effective_import_media or effective_import_embeddings:
                return False, "OpenWebUI imports do not use archive media or embedding restore options.", None
        if source_format_value == "openwebui_db" and not (selected_openwebui_user_id or "").strip():
            return False, "selected_openwebui_user_id is required for OpenWebUI DB imports", None

        try:
            if source_format_value in {"openwebui_json", "openwebui_db"}:
                resolved_path = self._resolve_import_upload_path(file_path)
            else:
                resolved_path = self._resolve_import_archive_path(file_path)
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"Chatbooks import rejected file path: {exc}")
            detail = (
                "Invalid or potentially malicious import file"
                if source_format_value in {"openwebui_json", "openwebui_db"}
                else "Invalid or potentially malicious archive file"
            )
            return False, detail, None
        file_token = self._build_import_file_token(resolved_path)
        safe_source_filename = Path(str(source_filename or "").replace("\\", "/")).name
        if safe_source_filename in {"", ".", ".."}:
            safe_source_filename = ""

        if not async_mode:
            self._check_chatbook_job_admission_with_lock("import")

        if async_mode:
            # Create job and run asynchronously
            job_id = str(uuid4())
            job = ImportJob(
                job_id=job_id,
                user_id=self.user_id,
                status=ImportStatus.PENDING,
                chatbook_path=file_token,
                metadata={
                    "source_format": source_format_value,
                    **(
                        {"source_filename": safe_source_filename}
                        if safe_source_filename
                        else {}
                    ),
                },
            )

            # Store job in database after atomically reserving Chatbooks quota.
            self._save_import_job_with_quota(job)

            # Start async task through core Jobs.
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
                    "content_selections": (
                        None
                        if content_selections is None
                        else {
                            k.value if hasattr(k, 'value') else str(k): v
                            for k, v in content_selections.items()
                        }
                    ),
                    "conflict_resolution": conflict_resolution.value if hasattr(conflict_resolution, 'value') else str(conflict_resolution),
                    "prefix_imported": bool(prefix_imported),
                    "import_media": effective_import_media,
                    "import_embeddings": effective_import_embeddings,
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
            sync_result = await asyncio.to_thread(
                self._import_chatbook_sync,
                str(resolved_path), content_selections,
                conflict_resolution, prefix_imported,
                effective_import_media, effective_import_embeddings
            )
            return await self.finalize_account_restore(*sync_result)

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
            import_warnings: list[str] = []

            if manifest.version == ChatbookVersion.V1_1:
                ok, v1_1_warnings, v1_1_errors = validate_v1_1_before_import(
                    manifest,
                    extract_dir,
                )
                import_warnings.extend(v1_1_warnings)
                if not ok:
                    first_error = v1_1_errors[0] if v1_1_errors else "validation failed"
                    return False, f"Chatbook v1.1 validation failed: {first_error}", {
                        "imported_items": {},
                        "warnings": import_warnings,
                        "errors": v1_1_errors,
                    }

            # Check version compatibility (V1 and V1_LEGACY are both compatible)
            compatible_versions = {ChatbookVersion.V1, ChatbookVersion.V1_LEGACY, ChatbookVersion.V1_1}
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
            import_status.warnings.extend(import_warnings)
            inventory_summary, skipped_non_restorable, inventory_warnings, inventory_errors = (
                self._inspect_manifest_inventory_for_import(manifest)
            )
            import_status.warnings.extend(inventory_warnings)
            if inventory_errors:
                return False, f"Chatbook inventory restore coverage missing: {inventory_errors[0]}", {
                    "imported_items": {},
                    "warnings": list(import_status.warnings or []),
                    "errors": inventory_errors,
                    "inventory_summary": inventory_summary,
                    "skipped_non_restorable": skipped_non_restorable,
                }

            account_restore_payload, account_payload_errors = self._load_account_restore_payloads(
                extract_dir,
                manifest,
            )
            if account_payload_errors:
                return False, f"Chatbook account restore payload invalid: {account_payload_errors[0]}", {
                    "imported_items": {},
                    "warnings": list(import_status.warnings or []),
                    "errors": account_payload_errors,
                    "inventory_summary": inventory_summary,
                    "skipped_non_restorable": skipped_non_restorable,
                }

            import_status.total_items = sum(len(ids) for ids in content_selections.values())

            supported_types = {
                ContentType.CHARACTER,
                ContentType.WORLD_BOOK,
                ContentType.DICTIONARY,
                ContentType.CONVERSATION,
                ContentType.NOTE,
                ContentType.EXPLAINER_SESSION,
                ContentType.GENERATED_DOCUMENT,
                ContentType.MEDIA,
                ContentType.EMBEDDING,
                ContentType.PROMPT,
                ContentType.EVALUATION,
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

            if not import_media and ContentType.MEDIA in content_selections:
                ids = content_selections.pop(ContentType.MEDIA) or []
                import_status.processed_items += len(ids)
                import_status.skipped_items += len(ids)
                import_status.warnings.append(
                    f"Skipped media restore because import_media=false ({len(ids)} items)"
                )
            if not import_embeddings and ContentType.EMBEDDING in content_selections:
                ids = content_selections.pop(ContentType.EMBEDDING) or []
                import_status.processed_items += len(ids)
                import_status.skipped_items += len(ids)
                import_status.warnings.append(
                    f"Skipped embedding restore because import_embeddings=false ({len(ids)} items)"
                )

            manifest_path_index = self._build_manifest_import_path_index(manifest)
            character_id_map: dict[str, int] = {}
            media_id_map: dict[str, int] = {}
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
                    manifest_path_index=manifest_path_index,
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
                    import_status,
                    manifest_path_index=manifest_path_index,
                )

            # Import dictionaries
            if ContentType.DICTIONARY in content_selections:
                _run_import_for_type(
                    ContentType.DICTIONARY,
                    self._import_dictionaries,
                    extract_dir, manifest,
                    content_selections[ContentType.DICTIONARY],
                    conflict_resolution, prefix_imported,
                    import_status,
                    manifest_path_index=manifest_path_index,
                )

            # Import prompts before dependent evaluation definitions.
            if ContentType.PROMPT in content_selections:
                _run_import_for_type(
                    ContentType.PROMPT,
                    self._import_prompts,
                    extract_dir, manifest,
                    content_selections[ContentType.PROMPT],
                    conflict_resolution, prefix_imported,
                    import_status,
                    manifest_path_index=manifest_path_index,
                )

            # Import media records before embeddings so media vector blobs can reattach.
            if ContentType.MEDIA in content_selections:
                _run_import_for_type(
                    ContentType.MEDIA,
                    self._import_media_items,
                    extract_dir, manifest,
                    content_selections[ContentType.MEDIA],
                    conflict_resolution, prefix_imported,
                    import_status,
                    manifest_path_index=manifest_path_index,
                    media_id_map=media_id_map,
                )

            # Import evaluations after prompts/media because archived definitions can reference both.
            if ContentType.EVALUATION in content_selections:
                _run_import_for_type(
                    ContentType.EVALUATION,
                    self._import_evaluations,
                    extract_dir, manifest,
                    content_selections[ContentType.EVALUATION],
                    conflict_resolution, prefix_imported,
                    import_status,
                    manifest_path_index=manifest_path_index,
                )

            if ContentType.EMBEDDING in content_selections:
                _run_import_for_type(
                    ContentType.EMBEDDING,
                    self._import_embeddings,
                    extract_dir, manifest,
                    content_selections[ContentType.EMBEDDING],
                    conflict_resolution, prefix_imported,
                    import_status,
                    manifest_path_index=manifest_path_index,
                    media_id_map=media_id_map,
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
                    manifest_path_index=manifest_path_index,
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
                    import_status,
                    manifest_path_index=manifest_path_index,
                )

            # Import Explainer sessions from first-class content items.
            if ContentType.EXPLAINER_SESSION in content_selections:
                _run_import_for_type(
                    ContentType.EXPLAINER_SESSION,
                    self._import_explainer_sessions,
                    extract_dir, manifest,
                    content_selections[ContentType.EXPLAINER_SESSION],
                    conflict_resolution, prefix_imported,
                    import_status
                )

            # Backward-compatible fallback: generated_document items with
            # metadata.subtype == "explainer_session" restore into Explainer.
            if ContentType.GENERATED_DOCUMENT in content_selections:
                explainer_document_ids: list[str] = []
                generic_document_ids: list[str] = []
                item_by_id = {
                    str(item.id): item
                    for item in manifest.content_items
                    if item.type == ContentType.GENERATED_DOCUMENT
                }
                for document_id in content_selections[ContentType.GENERATED_DOCUMENT]:
                    item = item_by_id.get(str(document_id))
                    metadata = item.metadata if item and isinstance(item.metadata, dict) else {}
                    if metadata.get("subtype") == "explainer_session":
                        explainer_document_ids.append(str(document_id))
                    else:
                        generic_document_ids.append(str(document_id))
                if explainer_document_ids:
                    _run_import_for_type(
                        ContentType.EXPLAINER_SESSION,
                        self._import_generated_document_explainer_sessions,
                        extract_dir, manifest,
                        explainer_document_ids,
                        conflict_resolution, prefix_imported,
                        import_status
                    )
                if generic_document_ids:
                    _run_import_for_type(
                        ContentType.GENERATED_DOCUMENT,
                        self._import_generated_documents,
                        extract_dir, manifest,
                        generic_document_ids,
                        conflict_resolution, prefix_imported,
                        import_status,
                        manifest_path_index=manifest_path_index,
                    )

            # Note: We do NOT delete the original import file - the caller owns it

            # Build result message
            logger.debug(f"Import status: total={import_status.total_items}, successful={import_status.successful_items}, skipped={import_status.skipped_items}, failed={import_status.failed_items}")
            sync_result = {
                "imported_items": imported_items,
                "warnings": list(import_status.warnings or []),
            }
            if inventory_summary:
                sync_result["inventory_summary"] = inventory_summary
            if skipped_non_restorable:
                sync_result["skipped_non_restorable"] = skipped_non_restorable
            if account_restore_payload:
                sync_result[_ACCOUNT_RESTORE_PAYLOAD_KEY] = account_restore_payload

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

    async def finalize_account_restore(
        self,
        success: bool,
        message: str,
        result: dict[str, Any] | None,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Apply private account payloads and remove them before results can be persisted."""
        if not isinstance(result, dict):
            return success, message, result
        account_payloads = result.pop(_ACCOUNT_RESTORE_PAYLOAD_KEY, None)
        if not success or not account_payloads:
            return success, message, result
        if not isinstance(account_payloads, dict):
            result.setdefault("errors", []).append("Account restore payload was invalid.")
            return False, "Account profile and settings restore failed", result

        try:
            restored = await self._restore_account_state_payloads(account_payloads)
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
            logger.error("Chatbooks account profile/settings restore failed; details redacted")
            result.setdefault("errors", []).append(
                "Account profile and settings could not be restored to the destination account."
            )
            return False, "Account profile and settings restore failed", result

        imported_items = result.setdefault("imported_items", {})
        for category, count in restored.items():
            imported_items[category] = int(count)
        restored_count = sum(restored.values())
        if restored_count:
            message = f"{message}; restored {restored_count} account profile/settings payload(s)"
        return True, message, result

    async def _restore_account_state_payloads(
        self,
        payloads: dict[str, dict[str, Any]],
    ) -> dict[str, int]:
        """Restore portable account state through AuthNZ and UserProfiles services."""
        if self.user_id_int is None:
            raise ValidationError("Account restore requires a numeric destination user id")

        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
        from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
        from tldw_Server_API.app.core.UserProfiles.command_service import ProfileCommandService
        from tldw_Server_API.app.core.UserProfiles.contracts import (
            ProfileContractMode,
            ProfileUpdateCommand,
        )

        db_pool = await get_db_pool()
        destination_user = await AuthnzUsersRepo(db_pool=db_pool).get_user_by_id(self.user_id_int)
        if destination_user is None:
            raise ValidationError("Destination account record does not exist")

        updates: list[tuple[str, Any]] = []
        profile_payload = payloads.get("account_profile")
        settings_payload = payloads.get("account_settings")
        if profile_payload:
            profile = profile_payload.get("profile") or {}
            if not isinstance(profile, dict):
                raise ValidationError("Account profile payload is invalid")
            email = profile.get("identity.email")
            destination_email = str(destination_user.get("email") or "").strip().casefold()
            restored_email = str(email or "").strip().casefold()
            if email is not None and restored_email != destination_email:
                updates.append(("identity.email", email))
        if settings_payload:
            overrides = settings_payload.get("overrides") or {}
            if not isinstance(overrides, dict):
                raise ValidationError("Account settings payload is invalid")
            updates.extend((str(key), value) for key, value in sorted(overrides.items()))

        if updates:
            command = ProfileUpdateCommand(
                actor_user_id=self.user_id_int,
                target_user_id=self.user_id_int,
                updates=tuple(updates),
                roles=frozenset({"user"}),
                dry_run=False,
                contract_mode=ProfileContractMode.LEGACY_V1,
            )
            async with db_pool.transaction() as db_conn:
                command_result = await ProfileCommandService(db_pool=db_pool).apply(
                    command,
                    db_conn=db_conn,
                    scope=None,
                )
            if command_result.status_code >= 400 or command_result.skipped:
                raise ValidationError("Destination rejected one or more account profile/settings values")

        return {
            "account_profile": 1 if profile_payload else 0,
            "account_settings": 1 if settings_payload else 0,
        }

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

            # Import chatbook synchronously using thread pool
            success, message, result = await asyncio.to_thread(
                self._import_chatbook_sync,
                file_path, content_selections,
                conflict_resolution, prefix_imported,
                import_media, import_embeddings
            )
            success, message, result = await self.finalize_account_restore(
                success,
                message,
                result,
            )

            if success:
                latest = self._get_import_job(job.job_id)
                if latest and latest.status == ImportStatus.CANCELLED:
                    return
                job.status = ImportStatus.COMPLETED
                if isinstance(result, dict):
                    imported_items = result.get("imported_items") if isinstance(result.get("imported_items"), dict) else {}
                    skipped_non_restorable = (
                        result.get("skipped_non_restorable")
                        if isinstance(result.get("skipped_non_restorable"), dict)
                        else {}
                    )
                    successful_count = sum(int(count or 0) for count in imported_items.values())
                    skipped_count = sum(int(count or 0) for count in skipped_non_restorable.values())
                    job.progress_percentage = 100
                    job.successful_items = successful_count
                    job.skipped_items = skipped_count
                    job.processed_items = successful_count + skipped_count
                    job.total_items = job.processed_items
                    result_warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
                    job.warnings = list(job.warnings or []) + [
                        warning for warning in result_warnings if warning not in (job.warnings or [])
                    ]
                    job.metadata = {
                        **(job.metadata if isinstance(job.metadata, dict) else {}),
                        "imported_items": imported_items,
                        "inventory_summary": result.get("inventory_summary"),
                        "skipped_non_restorable": skipped_non_restorable,
                    }
                else:
                    job.progress_percentage = 100
                    job.total_items = 0
                    job.processed_items = 0
            else:
                job.status = ImportStatus.FAILED
                job.error_message = message

            job.completed_at = datetime.now(timezone.utc)
            self._save_import_job(job)

        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            job.status = ImportStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error_message = str(e)
            self._save_import_job(job)
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

    def list_openwebui_import_scopes(self) -> list[dict[str, Any]]:
        """Return user-safe hydration scopes discovered from imported OpenWebUI conversations."""
        scopes: dict[str, dict[str, Any]] = {}
        client_id = getattr(self.db, "client_id", None)
        if client_id is None or not hasattr(self.db, "get_conversations_for_user"):
            return []

        offset = 0
        page_size = 500
        while True:
            conversations = self.db.get_conversations_for_user(
                client_id,
                limit=page_size,
                offset=offset,
            )
            if not conversations:
                break
            for conversation in conversations:
                conversation_id = str(conversation.get("id") or "").strip()
                if not conversation_id:
                    continue
                import_meta = self._openwebui_conversation_import_metadata(conversation_id)
                if not import_meta:
                    continue

                metadata = import_meta["metadata"]
                source_format = str(metadata.get("source_kind") or "openwebui_json").strip() or "openwebui_json"
                source_user_id = self._clean_optional_text(metadata.get("source_user_id"))
                source_user_label = self._clean_optional_text(metadata.get("source_user_label"))
                scope_id = self._openwebui_scope_id_from_metadata(metadata, source_format, source_user_id)
                created_at = self._clean_optional_text(metadata.get("imported_at"))
                attachment_count = self._count_openwebui_attachment_refs(conversation_id)
                source_conversation_id = (
                    self._clean_optional_text(metadata.get("row_id"))
                    or self._clean_optional_text(import_meta.get("external_ref"))
                    or self._clean_optional_text(conversation.get("external_ref"))
                )
                scope = scopes.setdefault(
                    scope_id,
                    {
                        "scope_id": scope_id,
                        "source_format": source_format,
                        "source_user_id": source_user_id,
                        "source_user_label": source_user_label,
                        "conversation_count": 0,
                        "attachment_reference_count": 0,
                        "created_at": created_at,
                        "conversation_ids": [],
                        "conversations": [],
                    },
                )
                if scope.get("created_at") is None and created_at is not None:
                    scope["created_at"] = created_at
                if scope.get("source_user_id") is None and source_user_id is not None:
                    scope["source_user_id"] = source_user_id
                if scope.get("source_user_label") is None and source_user_label is not None:
                    scope["source_user_label"] = source_user_label
                scope["conversation_ids"].append(conversation_id)
                scope["conversations"].append(
                    {
                        "source_conversation_id": source_conversation_id,
                        "conversation_id": conversation_id,
                        "title": str(conversation.get("title") or ""),
                        "attachment_reference_count": attachment_count,
                    }
                )
                scope["conversation_count"] += 1
                scope["attachment_reference_count"] += attachment_count

            if len(conversations) < page_size:
                break
            offset += page_size

        return sorted(
            scopes.values(),
            key=lambda item: str(item.get("created_at") or ""),
            reverse=True,
        )

    def resolve_openwebui_hydration_scope(self, scope: dict[str, Any]) -> dict[str, Any]:
        """Resolve an import-scope id to concrete tldw conversation ids for hydration."""
        import_scope_id = self._clean_optional_text(scope.get("import_scope_id"))
        source_user_id = self._clean_optional_text(scope.get("source_user_id") or scope.get("openwebui_user_id"))
        conversation_ids = [
            text
            for text in (self._clean_optional_text(item) for item in (scope.get("conversation_ids") or []))
            if text is not None
        ]
        if conversation_ids:
            return {
                "import_scope_id": import_scope_id,
                "conversation_ids": conversation_ids,
                "source_user_id": source_user_id,
            }
        if import_scope_id:
            for imported_scope in self.list_openwebui_import_scopes():
                if imported_scope["scope_id"] != import_scope_id:
                    continue
                return {
                    "import_scope_id": import_scope_id,
                    "conversation_ids": list(imported_scope.get("conversation_ids") or []),
                    "source_user_id": source_user_id or imported_scope.get("source_user_id"),
                }
            raise ValueError("OpenWebUI import scope was not found.")
        return {
            "import_scope_id": None,
            "conversation_ids": [],
            "source_user_id": source_user_id,
        }

    def preview_openwebui_attachment_hydration(
        self,
        *,
        openwebui_data_root: str,
        scope: dict[str, Any],
        process_supported_files: bool = False,
    ) -> dict[str, Any]:
        """Preview OpenWebUI attachment hydration for already imported conversations."""
        resolved_scope = self.resolve_openwebui_hydration_scope(scope)
        source_user_id = resolved_scope.get("source_user_id")
        conversation_ids = tuple(resolved_scope.get("conversation_ids") or [])
        data_root = validate_openwebui_data_root(openwebui_data_root, require_uploads=False)

        items: list[dict[str, Any]] = []
        warnings: list[str] = []
        with open_validated_openwebui_db(data_root.webui_db_path) as conn:
            validate_openwebui_file_schema(conn)
            preview = extract_openwebui_hydration_references(
                self.db,
                OpenWebUIHydrationScope(
                    conversation_ids=conversation_ids,
                    openwebui_user_id=str(source_user_id).strip() if source_user_id else None,
                ),
                openwebui_conn=conn,
            )
            items.extend(self._openwebui_hydration_item_to_dict(item) for item in preview.items)
            warnings.extend(str(warning) for warning in preview.warnings)

            file_rows = load_openwebui_file_rows_for_ids(
                conn,
                tuple(reference.file_id for reference in preview.references),
                str(source_user_id).strip() if source_user_id else None,
            )
            rows_by_id = {str(row["id"]): row for row in file_rows}
            for reference in preview.references:
                file_row = rows_by_id.get(reference.file_id)
                if file_row is None:
                    items.append(
                        {
                            "conversation_id": reference.conversation_id,
                            "message_id": reference.message_id,
                            "file_id": reference.file_id,
                            "status": "missing_source_file_row",
                            "warning_code": "missing_source_file_row",
                            "raw_ref_index": reference.raw_ref_index,
                            "source": reference.source,
                        }
                    )
                    continue

                resolved = resolve_openwebui_file_path(file_row, data_root)
                warning_code = resolved.warning_codes[0] if resolved.warning_codes else None
                items.append(
                    {
                        "conversation_id": reference.conversation_id,
                        "message_id": reference.message_id,
                        "file_id": reference.file_id,
                        "status": resolved.status,
                        "warning_code": warning_code,
                        "raw_ref_index": reference.raw_ref_index,
                        "source": reference.source,
                        "file_kind": resolved.file_kind,
                        "mime_type": resolved.mime_type,
                    }
                )

        summary = self._openwebui_hydration_summary(items)
        summary["referenced_files"] = len(items)
        summary["warning_count"] = len(warnings) + sum(1 for item in items if item.get("warning_code"))
        returned_items, omitted_items = self._cap_openwebui_hydration_items(items)
        summary["returned_items"] = len(returned_items)
        summary["omitted_items"] = omitted_items
        return {
            "scope": resolved_scope,
            "process_supported_files": bool(process_supported_files),
            "summary": summary,
            "items": returned_items,
            "warnings": warnings,
        }

    def run_openwebui_attachment_hydration(
        self,
        *,
        openwebui_data_root: str,
        scope: dict[str, Any],
        process_supported_files: bool = False,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Hydrate OpenWebUI attachments into tldw message images and Media DB records."""
        if self.user_id_int is None:
            raise ValueError("OpenWebUI attachment hydration requires a numeric user id.")

        resolved_scope = self.resolve_openwebui_hydration_scope(scope)
        source_user_id = resolved_scope.get("source_user_id")
        conversation_ids = tuple(resolved_scope.get("conversation_ids") or [])
        data_root = validate_openwebui_data_root(openwebui_data_root, require_uploads=True)
        media_db = self._get_media_db()
        storage_root = self._openwebui_attachment_storage_root()
        run_dedupe_cache: dict[tuple[str, str], int] = {}

        items: list[dict[str, Any]] = []
        warnings: list[str] = []
        resolved_files = 0
        image_files = 0
        media_files = 0

        with open_validated_openwebui_db(data_root.webui_db_path) as conn:
            validate_openwebui_file_schema(conn)
            preview = extract_openwebui_hydration_references(
                self.db,
                OpenWebUIHydrationScope(
                    conversation_ids=conversation_ids,
                    openwebui_user_id=str(source_user_id).strip() if source_user_id else None,
                ),
                openwebui_conn=conn,
            )
            items.extend(self._openwebui_hydration_item_to_dict(item) for item in preview.items)
            warnings.extend(str(warning) for warning in preview.warnings)

            file_rows = load_openwebui_file_rows_for_ids(
                conn,
                tuple(reference.file_id for reference in preview.references),
                str(source_user_id).strip() if source_user_id else None,
            )
            rows_by_id = {str(row["id"]): row for row in file_rows}
            for reference in preview.references:
                file_row = rows_by_id.get(reference.file_id)
                if file_row is None:
                    items.append(
                        {
                            "conversation_id": reference.conversation_id,
                            "message_id": reference.message_id,
                            "file_id": reference.file_id,
                            "status": "missing_source_file_row",
                            "warning_code": "missing_source_file_row",
                            "raw_ref_index": reference.raw_ref_index,
                            "source": reference.source,
                        }
                    )
                    continue

                resolved = resolve_openwebui_file_path(file_row, data_root)
                if resolved.status != "resolved":
                    warning_code = resolved.warning_codes[0] if resolved.warning_codes else resolved.status
                    items.append(
                        {
                            "conversation_id": reference.conversation_id,
                            "message_id": reference.message_id,
                            "file_id": reference.file_id,
                            "status": resolved.status,
                            "warning_code": warning_code,
                            "raw_ref_index": reference.raw_ref_index,
                            "source": reference.source,
                            "file_kind": resolved.file_kind,
                            "mime_type": resolved.mime_type,
                        }
                    )
                    continue

                resolved_files += 1
                if resolved.file_kind == "image":
                    image_files += 1
                    item = hydrate_image_reference(
                        self.db,
                        reference,
                        resolved,
                        job_id=job_id,
                    )
                elif media_db is None:
                    media_files += 1
                    items.append(
                        {
                            "conversation_id": reference.conversation_id,
                            "message_id": reference.message_id,
                            "file_id": reference.file_id,
                            "status": "media_db_unavailable",
                            "warning_code": "media_db_unavailable",
                            "raw_ref_index": reference.raw_ref_index,
                            "source": reference.source,
                            "file_kind": resolved.file_kind,
                            "mime_type": resolved.mime_type,
                        }
                    )
                    continue
                else:
                    media_files += 1
                    item = register_non_image_reference(
                        self.db,
                        media_db,
                        reference,
                        resolved,
                        owner_user_id=self.user_id_int,
                        storage_root=storage_root,
                        job_id=job_id,
                        process_supported_files=bool(process_supported_files),
                        run_dedupe_cache=run_dedupe_cache,
                    )
                items.append(self._openwebui_hydration_item_to_dict(item))

        summary = self._openwebui_hydration_execution_summary(
            items,
            resolved_files=resolved_files,
            image_files=image_files,
            media_files=media_files,
        )
        item_warnings = [
            f"{item.get('file_id') or 'unknown'}:{item.get('warning_code')}"
            for item in items
            if item.get("warning_code")
        ]
        total_warning_count = len(warnings) + len(item_warnings)
        warnings.extend(item_warnings[:MAX_PREVIEW_WARNING_ITEMS])
        summary["warning_count"] = total_warning_count
        returned_items, omitted_items = self._cap_openwebui_hydration_items(items)
        summary["returned_items"] = len(returned_items)
        summary["omitted_items"] = omitted_items
        return {
            "scope": resolved_scope,
            "process_supported_files": bool(process_supported_files),
            "summary": summary,
            "items": returned_items,
            "warnings": warnings,
        }

    def _openwebui_attachment_storage_root(self) -> Path:
        """Return the tldw-owned storage root for hydrated non-image attachments."""
        raw_path = (
            os.getenv("OPENWEBUI_HYDRATION_MEDIA_STORAGE_PATH")
            or os.getenv("MEDIA_STORAGE_PATH")
            or ""
        ).strip()
        if raw_path:
            storage_root = Path(raw_path).expanduser()
            if not storage_root.is_absolute():
                storage_root = (ACTUAL_PROJECT_ROOT / storage_root).resolve(strict=False)
        else:
            storage_root = ACTUAL_PROJECT_ROOT / "Databases" / "media_storage"
        storage_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with contextlib.suppress(OSError):
            storage_root.chmod(0o700)
        return storage_root

    @staticmethod
    def _clean_optional_text(value: Any) -> str | None:
        """Return stripped text, or None for blank/null values."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _openwebui_scope_id_from_metadata(
        metadata: dict[str, Any],
        source_format: str,
        source_user_id: str | None,
    ) -> str:
        """Return stored scope id, with a deterministic legacy fallback."""
        stored = ChatbookService._clean_optional_text(metadata.get("import_scope_id"))
        if stored:
            return stored
        user_part = source_user_id or "all"
        return f"{source_format}:{user_part}"

    def _openwebui_conversation_import_metadata(self, conversation_id: str) -> dict[str, Any] | None:
        """Load OpenWebUI conversation import settings in a normalized envelope."""
        if not hasattr(self.db, "get_conversation_settings"):
            return None
        settings_row = self.db.get_conversation_settings(conversation_id)
        settings = settings_row.get("settings") if isinstance(settings_row, dict) else None
        if not isinstance(settings, dict):
            return None
        openwebui_import = settings.get("openwebui_import")
        if not isinstance(openwebui_import, dict):
            return None
        metadata = openwebui_import.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            **openwebui_import,
            "metadata": metadata,
        }

    def _count_openwebui_attachment_refs(self, conversation_id: str) -> int:
        """Count preserved OpenWebUI attachment references for one imported conversation."""
        count = 0
        offset = 0
        page_size = 500
        while True:
            messages = self.db.get_messages_for_conversation(
                conversation_id,
                limit=page_size,
                offset=offset,
            )
            if not messages:
                break
            message_ids = [str(message["id"]) for message in messages if message.get("id") is not None]
            metadata_by_message_id = self.db.get_message_metadata_map(message_ids)
            for message_id in message_ids:
                metadata = metadata_by_message_id.get(message_id)
                extra = metadata.get("extra") if isinstance(metadata, dict) else None
                openwebui_import = extra.get("openwebui_import") if isinstance(extra, dict) else None
                refs = openwebui_import.get("attachment_refs") if isinstance(openwebui_import, dict) else None
                if isinstance(refs, list):
                    count += len(refs)
            if len(messages) < page_size:
                break
            offset += page_size
        return count

    @staticmethod
    def _openwebui_hydration_item_to_dict(item: Any) -> dict[str, Any]:
        """Convert a hydration preview item to a raw-path-free dict."""
        return {
            "conversation_id": item.conversation_id,
            "message_id": item.message_id,
            "file_id": item.file_id,
            "status": item.status,
            "warning_code": item.warning_code,
            "raw_ref_index": item.raw_ref_index,
            "source": item.source,
            "raw_ref_shape": item.raw_ref_shape,
            "job_id": item.job_id,
            "source_key": item.source_key,
            "message_image_position": item.message_image_position,
            "mime_type": item.mime_type,
            "media_id": item.media_id,
            "media_file_id": item.media_file_id,
            "checksum": item.checksum,
            "processing_status": item.processing_status,
        }

    @staticmethod
    def _openwebui_hydration_summary(items: list[dict[str, Any]]) -> dict[str, int]:
        """Build user-facing hydration preview counts."""
        resolved = [item for item in items if item.get("status") == "resolved"]
        return {
            "referenced_files": len(items),
            "returned_items": len(items),
            "omitted_items": 0,
            "resolved_files": len(resolved),
            "image_files": sum(1 for item in resolved if item.get("file_kind") == "image"),
            "media_files": sum(1 for item in resolved if item.get("file_kind") != "image"),
            "missing_files": sum(
                1 for item in items if item.get("status") in {"missing_file", "missing_source_file_row"}
            ),
            "unsupported_files": sum(
                1
                for item in items
                if item.get("status") in {"unsupported_file_type", "unsupported_reference_shape"}
            ),
            "failed_files": sum(1 for item in items if item.get("status") in {"path_rejected"}),
            "hydrated_images": 0,
            "registered_media_files": 0,
            "already_hydrated": 0,
            "processed_files": 0,
            "warning_count": 0,
        }

    @staticmethod
    def _openwebui_hydration_execution_summary(
        items: list[dict[str, Any]],
        *,
        resolved_files: int,
        image_files: int,
        media_files: int,
    ) -> dict[str, int]:
        """Build final hydration execution counts."""
        statuses = [str(item.get("status") or "") for item in items]
        failed_statuses = {
            "media_db_unavailable",
            "media_registration_failed",
            "metadata_update_failed",
            "message_missing",
            "oversized",
            "path_rejected",
        }
        return {
            "referenced_files": len(items),
            "returned_items": len(items),
            "omitted_items": 0,
            "resolved_files": resolved_files,
            "image_files": image_files,
            "media_files": media_files,
            "missing_files": sum(
                1 for status in statuses if status in {"missing_file", "missing_source_file_row"}
            ),
            "unsupported_files": sum(
                1
                for status in statuses
                if status in {"unsupported_file_type", "unsupported_reference_shape"}
            ),
            "failed_files": sum(1 for status in statuses if status in failed_statuses),
            "hydrated_images": statuses.count("hydrated_image"),
            "registered_media_files": statuses.count("registered_media"),
            "already_hydrated": statuses.count("already_hydrated") + statuses.count("already_registered_media"),
            "processed_files": sum(1 for item in items if item.get("processing_status") == "completed"),
            "warning_count": 0,
        }

    @staticmethod
    def _cap_openwebui_hydration_items(
        items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """Return a bounded hydration item list and omitted-item count."""
        limit = max(0, int(MAX_OPENWEBUI_HYDRATION_RESPONSE_ITEMS))
        if len(items) <= limit:
            return list(items), 0
        return list(items[:limit]), len(items) - limit

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
        metadata = dict(chat.source_metadata)
        if extra_metadata:
            metadata.update(extra_metadata)

        def merge(settings: dict[str, Any]) -> dict[str, Any]:
            merged = dict(settings)
            merged["openwebui_import"] = {
                "source": "openwebui",
                "external_ref": external_ref,
                "history_current_id": chat.history_current_id,
                "branched": chat.is_branched,
                "metadata": metadata,
            }
            return merged

        return self._persist_openwebui_settings_mutation(
            conversation_id,
            merge,
            operation="conversation metadata",
        )

    def _persist_openwebui_settings_mutation(
        self,
        conversation_id: str,
        merge: Callable[[dict[str, Any]], dict[str, Any] | None],
        *,
        operation: str,
    ) -> bool:
        """Lock, validate, and CAS one OpenWebUI settings mutation."""
        try:
            with self.db.transaction() as conn:
                resume_state = self.db.get_roleplay_resume_state(
                    conversation_id,
                    conn=conn,
                    lock_for_update=True,
                )
                settings = resume_state.get("settings")
                merged = merge(dict(settings) if isinstance(settings, dict) else {})
                if merged is None:
                    return False
                conversation = resume_state.get("conversation")
                if not isinstance(conversation, dict):
                    return False
                character_id = conversation.get("character_id")
                snapshot = resume_state.get("behavior_snapshot")
                snapshot_valid = (
                    isinstance(snapshot, dict) and snapshot.get("status") == "valid"
                )
                validated = validate_chat_settings_storage(
                    merged,
                    reject_credentials=snapshot_valid,
                    allow_internal=True,
                    behavior_snapshot=snapshot,
                    conversation={"character_id": character_id},
                )
                settings_version = resume_state.get("settings_version")
                return bool(
                    self.db.upsert_conversation_settings(
                        conversation_id,
                        validated,
                        conn=conn,
                        expected_settings_version=(
                            settings_version if isinstance(settings_version, int) else 0
                        ),
                    )
                )
        except (CharactersRAGDBError, ValueError) as exc:
            logger.warning(
                "OpenWebUI {} settings write failed for {}: {}",
                operation,
                conversation_id,
                exc,
            )
            return False

    def _record_openwebui_import_mapping(
        self,
        conversation_id: str,
        *,
        import_scope_id: str,
        source_format: str,
        source_user_id: str | None = None,
        source_user_label: str | None = None,
        imported_at: str | None = None,
    ) -> bool:
        """Persist hydration-scope metadata for an imported OpenWebUI conversation."""
        def merge(settings: dict[str, Any]) -> dict[str, Any] | None:
            openwebui_import = settings.get("openwebui_import")
            if not isinstance(openwebui_import, dict):
                return None
            metadata = openwebui_import.get("metadata")
            merged_metadata = dict(metadata) if isinstance(metadata, dict) else {}
            merged_metadata["import_scope_id"] = str(import_scope_id)
            merged_metadata["source_kind"] = str(source_format)
            merged_metadata["imported_at"] = imported_at or datetime.now(timezone.utc).isoformat()
            if source_user_id is not None:
                merged_metadata["source_user_id"] = str(source_user_id)
            if source_user_label is not None:
                merged_metadata["source_user_label"] = str(source_user_label)

            merged_openwebui_import = dict(openwebui_import)
            merged_openwebui_import["metadata"] = merged_metadata
            merged_settings = dict(settings)
            merged_settings["openwebui_import"] = merged_openwebui_import
            return merged_settings

        return self._persist_openwebui_settings_mutation(
            conversation_id,
            merge,
            operation="import mapping",
        )

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
        import_scope_id = f"openwebui_json:{uuid4().hex}"
        imported_at = datetime.now(timezone.utc).isoformat()

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
                if not self._record_openwebui_import_mapping(
                    conversation_id,
                    import_scope_id=import_scope_id,
                    source_format="openwebui_json",
                    imported_at=imported_at,
                ):
                    raise DatabaseError(
                        f"OpenWebUI chat {chat.external_ref} import mapping was not stored."
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
        import_scope_id = f"openwebui_db:{extracted.selected_user_id}:{uuid4().hex}"
        imported_at = datetime.now(timezone.utc).isoformat()
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
                if not self._record_openwebui_import_mapping(
                    conversation_id,
                    import_scope_id=import_scope_id,
                    source_format="openwebui_db",
                    source_user_id=extracted.selected_user_id,
                    source_user_label=extracted.selected_user_label,
                    imported_at=imported_at,
                ):
                    raise DatabaseError(
                        f"OpenWebUI chat {chat.external_ref} import mapping was not stored."
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
                        owner_user_id=self.user_id,
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
        manifest, error, _report = self._preview_chatbook_internal(file_path, include_report=False)
        return manifest, error

    def preview_chatbook_with_report(
        self,
        file_path: str,
    ) -> tuple[ChatbookManifest | None, str | None, dict[str, Any] | None]:
        """
        Preview a chatbook and return a v1.1 report when the manifest loads.

        The original preview_chatbook two-tuple is intentionally preserved for
        existing callers.
        """
        return self._preview_chatbook_internal(file_path, include_report=True)

    def _preview_chatbook_internal(
        self,
        file_path: str,
        *,
        include_report: bool,
    ) -> tuple[ChatbookManifest | None, str | None, dict[str, Any] | None]:
        """Shared safe extraction flow for chatbook preview variants."""
        extract_dir: Path | None = None
        try:
            try:
                resolved_path = self._resolve_import_archive_path(file_path)
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"Chatbooks preview rejected file path: {exc}")
                return None, "Invalid or potentially malicious archive file", None
            file_path = str(resolved_path)

            # Defense-in-depth: validate the archive before extraction.
            # Any unexpected validator fault should surface as a server error.
            from .chatbook_validators import ChatbookValidator
            ok, err = ChatbookValidator.validate_zip_file(file_path)
            if not ok:
                return None, err or "Invalid archive", None
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
                        return None, "Unsafe path in archive detected", None

                    # Additional check: ensure the normalized target stays within extract_dir
                    os.path.normpath(member)
                    # Use normpath + commonpath instead of realpath to avoid race conditions
                    target_path = os.path.normpath(os.path.join(extract_dir_resolved, member))
                    try:
                        common = os.path.commonpath([extract_dir_resolved, target_path])
                        if common != extract_dir_resolved:
                            return None, "Path traversal attempt detected", None
                    except ValueError:
                        # commonpath raises ValueError if paths are on different drives (Windows)
                        return None, "Path traversal attempt detected", None

                # Safe to extract after validation
                zf.extractall(extract_dir)

            # Load manifest
            manifest_path = extract_dir / "manifest.json"
            if not manifest_path.exists():
                return None, "Invalid chatbook: manifest.json not found", None

            try:
                with open(manifest_path, encoding='utf-8') as f:
                    manifest_data = json.load(f)
                manifest = ChatbookManifest.from_dict(manifest_data)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, ValidationError):
                return None, "Invalid chatbook manifest", None

            report = build_preview_report(manifest, extract_dir) if include_report else None
            return manifest, None, report

        except zipfile.BadZipFile:
            return None, "Invalid archive", None
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
        if job and getattr(self, "_jobs_adapter", None) is not None:
            with contextlib.suppress(_CHATBOOK_NONCRITICAL_EXCEPTIONS):
                self._jobs_adapter.apply_export_status(job)
        if job:
            self._normalize_job_timestamps_for_api(job)
        return job

    def get_import_job(self, job_id: str) -> ImportJob | None:
        """Get import job status."""
        job = self._get_import_job(job_id)
        if job and getattr(self, "_jobs_adapter", None) is not None:
            with contextlib.suppress(_CHATBOOK_NONCRITICAL_EXCEPTIONS):
                self._jobs_adapter.apply_import_status(job)
        if job:
            self._normalize_job_timestamps_for_api(job)
        return job

    def list_export_jobs(
        self,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
        *,
        raise_on_error: bool = False,
    ) -> list[ExportJob]:
        """List all export jobs for this user.

        Args:
            status: Optional status filter (pending, in_progress, completed, failed, cancelled, expired)
            limit: Maximum number of jobs to return
            offset: Offset for pagination
            raise_on_error: Propagate database errors for destructive workflows
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
                    metadata = {}
                    raw_metadata = row.get("metadata")
                    if isinstance(raw_metadata, str):
                        with contextlib.suppress(json.JSONDecodeError):
                            metadata = json.loads(raw_metadata)
                    elif isinstance(raw_metadata, dict):
                        metadata = raw_metadata
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
                        metadata=metadata
                    )
                else:
                    # Handle tuple format from mocked tests
                    # (job_id, user_id, status, chatbook_name, output_path, created_at,
                    #  started_at, completed_at, error_message, progress_percentage,
                    #  total_items, processed_items, file_size_bytes, download_url, expires_at)
                    raw_metadata = row[15] if len(row) > 15 else None
                    metadata = {}
                    if isinstance(raw_metadata, str):
                        with contextlib.suppress(json.JSONDecodeError):
                            metadata = json.loads(raw_metadata)
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
                        metadata=metadata
                    )
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

            for job in jobs:
                self._normalize_job_timestamps_for_api(job)

            return jobs
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error listing export jobs: {e}")
            if raise_on_error:
                raise
            return []

    def list_import_jobs(
        self,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
        *,
        raise_on_error: bool = False,
    ) -> list[ImportJob]:
        """List all import jobs for this user.

        Args:
            status: Optional status filter (pending, validating, in_progress, completed, failed, cancelled)
            limit: Maximum number of jobs to return
            offset: Offset for pagination
            raise_on_error: Propagate database errors for destructive workflows
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
                    metadata = {}
                    raw_metadata = row.get("metadata")
                    if isinstance(raw_metadata, str):
                        with contextlib.suppress(json.JSONDecodeError):
                            metadata = json.loads(raw_metadata)
                    elif isinstance(raw_metadata, dict):
                        metadata = raw_metadata
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
                        warnings=json.loads(row['warnings']) if row['warnings'] else [],
                        metadata=metadata,
                    )
                else:
                    # Handle tuple format from mocked tests
                    # (job_id, user_id, status, chatbook_path, created_at, started_at,
                    #  completed_at, error_message, progress_percentage, total_items,
                    #  processed_items, successful_items, failed_items, skipped_items,
                    #  conflicts, warnings, metadata)
                    metadata = {}
                    raw_metadata = row[16] if len(row) > 16 else None
                    if isinstance(raw_metadata, str):
                        with contextlib.suppress(json.JSONDecodeError):
                            metadata = json.loads(raw_metadata)
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
                        warnings=json.loads(row[15]) if len(row) > 15 and row[15] else [],
                        metadata=metadata,
                    )
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

            for job in jobs:
                self._normalize_job_timestamps_for_api(job)

            return jobs
        except _CHATBOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error listing import jobs: {e}")
            if raise_on_error:
                raise
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

            chunks: list[dict[str, Any]] = []
            if get_unvectorized_chunk_count is not None and get_unvectorized_chunks_in_range is not None:
                try:
                    chunk_count = get_unvectorized_chunk_count(media_db, int(media_record["id"])) or 0
                    if chunk_count:
                        chunks = [
                            self._convert_datetimes(row)
                            for row in get_unvectorized_chunks_in_range(media_db, int(media_record["id"]), 0, chunk_count)
                        ]
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(f"Failed to fetch media chunks for media {media_id}: {exc}")
                    self._note_todo("Media chunks export failed for some items; inspect logs.")
            normalized["chunks"] = chunks

            media_files: list[dict[str, Any]] = []
            bundled_file_count = 0
            pointer_file_count = 0
            if include_media:
                media_files, bundled_file_count, pointer_file_count = self._copy_media_artifacts(
                    media_db,
                    int(media_record["id"]),
                    media_dir,
                    media_id,
                )
            normalized["stored_artifacts"] = media_files
            if pointer_file_count:
                pointer_warnings = manifest.metadata.setdefault("pointer_only_warnings", []) if manifest.metadata is not None else []
                pointer_warnings.append(
                    f"{pointer_file_count} stored media artifact pointer(s) for media {media_id} did not resolve under account storage."
                )

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
                file_path=f"content/media/{media_file.name}",
                metadata={
                    "transcript_count": len(transcripts),
                    "chunk_count": len(chunks),
                    "stored_artifact_count": bundled_file_count,
                    "pointer_only_artifact_count": pointer_file_count,
                },
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
                    embeddings_data = result.get("embeddings")
                    if hasattr(embeddings_data, "tolist"):
                        embeddings_data = embeddings_data.tolist()
                    if embeddings_data is None:
                        embeddings_data = []

                    for i, chunk_id in enumerate(ids):
                        if max_chunks_per_collection > 0 and len(chunks) >= max_chunks_per_collection:
                            truncated = True
                            break
                        chunk: dict[str, Any] = {"id": chunk_id}
                        if documents and i < len(documents):
                            chunk["document"] = documents[i]
                        if metadatas and i < len(metadatas):
                            chunk["metadata"] = metadatas[i]
                        if i < len(embeddings_data):
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
        if not world_book_ids:
            return
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
        if not dictionary_ids:
            return
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
        if not document_ids:
            return
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

    def _collect_explainer_sessions(
        self,
        session_ids: list[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent,
    ) -> None:
        """Collect complete Explainer sessions as first-class Chatbook content."""
        if not session_ids:
            return
        sessions_dir = work_dir / "content" / "explainer_sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        repo = self._get_explainer_repo()

        for session_id in session_ids:
            session_id_text = str(session_id)
            payload = build_explainer_chatbook_payload(
                repo=repo,
                session_id=session_id_text,
                owner_user_id=self.user_id,
            )
            session_data = payload["structured"]["session"]
            session_file = sessions_dir / f"session_{session_id_text}.json"
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            content.explainer_sessions[session_id_text] = payload
            manifest.content_items.append(
                ContentItem(
                    id=session_id_text,
                    type=ContentType.EXPLAINER_SESSION,
                    title=session_data.get("title") or "Explainer session",
                    created_at=self._parse_timestamp(session_data.get("createdAt")),
                    updated_at=self._parse_timestamp(session_data.get("updatedAt")),
                    file_path=f"content/explainer_sessions/session_{session_id_text}.json",
                    metadata={"format": EXPLAINER_CHATBOOK_FORMAT},
                )
            )

    # Helper methods for importing content

    def _import_conversations(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        conversation_ids: list[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob,
        manifest_path_index: ManifestImportPathIndex | None = None,
        character_id_map: dict[str, int] | None = None
    ):
        """Import conversations from chatbook."""
        max_img_bytes = self._get_max_message_image_bytes()
        if manifest_path_index is None:
            manifest_path_index = self._build_manifest_import_path_index(manifest)

        for conv_id in conversation_ids:
            status.processed_items += 1

            try:
                # Load conversation file
                conv_file, conv_path = self._resolve_manifest_import_file(
                    extract_dir,
                    manifest,
                    ContentType.CONVERSATION,
                    conv_id,
                    manifest_path_index,
                )
                if conv_file is None:
                    status.failed_items += 1
                    status.warnings.append(f"Unsafe conversation file path: {conv_path}")
                    continue
                if not conv_file.exists():
                    status.failed_items += 1
                    status.warnings.append(f"Conversation file not found: {conv_path}")
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
        status: ImportJob,
        manifest_path_index: ManifestImportPathIndex | None = None,
    ):
        """Import notes from chatbook."""
        template_settings = self._resolve_template_settings(manifest)
        if manifest_path_index is None:
            manifest_path_index = self._build_manifest_import_path_index(manifest)

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
                note_file, note_path = self._resolve_manifest_import_file(
                    extract_dir,
                    manifest,
                    ContentType.NOTE,
                    note_id,
                    manifest_path_index,
                )
                if note_file is None:
                    status.failed_items += 1
                    status.warnings.append(f"Unsafe note file path: {note_path}")
                    continue
                if not note_file.exists():
                    status.failed_items += 1
                    status.warnings.append(f"Note file not found: {note_path}")
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

                coordinator = active_coordinator(self.db, user_id=self.user_id)
                if coordinator is not None:
                    key = coordinator.request_fingerprint(
                        "chatbook.note.import",
                        {
                            "source_note_id": note_id,
                            "title": note_title,
                            "content": note_content,
                        },
                    )
                    new_note_id = stable_note_id("chatbook-import", key)
                    capture_note_upsert(
                        coordinator,
                        note_id=new_note_id,
                        title=note_title,
                        content=note_content,
                        source="chatbook-import",
                        key=key,
                    )
                else:
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

    def _import_explainer_sessions(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        session_ids: list[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob,
    ) -> None:
        """Import first-class Explainer session payloads from a chatbook."""
        item_by_id = {
            item.id: item
            for item in manifest.content_items
            if item.type == ContentType.EXPLAINER_SESSION
        }
        sessions_dir = extract_dir / "content" / "explainer_sessions"
        repo = self._get_explainer_repo()

        for session_id in session_ids:
            session_id_text = str(session_id)
            status.processed_items += 1
            item = item_by_id.get(session_id_text)
            rel_path = item.file_path if item and item.file_path else f"content/explainer_sessions/session_{session_id_text}.json"
            payload_path = self._safe_extracted_content_path(extract_dir, rel_path)
            if payload_path is None and sessions_dir.exists():
                payload_path = sessions_dir / f"session_{session_id_text}.json"
            if payload_path is None or not payload_path.exists():
                status.failed_items += 1
                status.warnings.append(f"Explainer session file not found: session_{session_id_text}.json")
                continue

            try:
                with open(payload_path, encoding="utf-8") as f:
                    payload = json.load(f)
                restore_explainer_chatbook_payload(
                    repo=repo,
                    payload=payload,
                    owner_user_id=self.user_id,
                    prefix_imported=prefix_imported,
                )
                status.successful_items += 1
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                status.failed_items += 1
                status.warnings.append(f"Error importing Explainer session {session_id_text}: {str(exc)}")

    def _import_generated_document_explainer_sessions(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        document_ids: list[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob,
    ) -> None:
        """Restore generated_document fallback items that carry Explainer sessions."""
        item_by_id = {
            item.id: item
            for item in manifest.content_items
            if item.type == ContentType.GENERATED_DOCUMENT
        }
        repo = self._get_explainer_repo()

        for document_id in document_ids:
            document_id_text = str(document_id)
            status.processed_items += 1
            item = item_by_id.get(document_id_text)
            metadata = item.metadata if item and isinstance(item.metadata, dict) else {}
            if metadata.get("subtype") != "explainer_session":
                status.skipped_items += 1
                status.warnings.append(
                    f"Skipped generated document {document_id_text}: unsupported generated_document subtype"
                )
                continue

            rel_path = item.file_path if item and item.file_path else f"content/generated_documents/document_{document_id_text}.json"
            payload_path = self._safe_extracted_content_path(extract_dir, rel_path)
            if payload_path is None or not payload_path.exists():
                status.failed_items += 1
                status.warnings.append(f"Generated document file not found: document_{document_id_text}.json")
                continue

            try:
                with open(payload_path, encoding="utf-8") as f:
                    payload = json.load(f)
                if not isinstance(payload, dict):
                    raise ValueError("generated document payload must be an object")
                payload.setdefault("type", "generated_document")
                payload.setdefault("metadata", metadata)
                restore_explainer_chatbook_payload(
                    repo=repo,
                    payload=payload,
                    owner_user_id=self.user_id,
                    prefix_imported=prefix_imported,
                )
                status.successful_items += 1
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                status.failed_items += 1
                status.warnings.append(
                    f"Error importing generated Explainer document {document_id_text}: {str(exc)}"
                )

    def _import_prompts(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        prompt_ids: list[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob,
        manifest_path_index: ManifestImportPathIndex | None = None,
    ) -> None:
        """Import Prompt DB rows from a chatbook."""
        prompts_db = self._get_prompts_db()
        if manifest_path_index is None:
            manifest_path_index = self._build_manifest_import_path_index(manifest)

        for prompt_id in prompt_ids:
            prompt_id_text = str(prompt_id)
            status.processed_items += 1
            if prompts_db is None:
                status.failed_items += 1
                status.warnings.append("Prompt import failed: prompts database is unavailable.")
                continue
            prompt_file, prompt_path = self._resolve_manifest_import_file(
                extract_dir,
                manifest,
                ContentType.PROMPT,
                prompt_id_text,
                manifest_path_index,
            )
            if prompt_file is None:
                status.failed_items += 1
                status.warnings.append(f"Unsafe prompt file path: {prompt_path}")
                continue
            if not prompt_file.exists():
                status.failed_items += 1
                status.warnings.append(f"Prompt file not found: {prompt_path}")
                continue

            try:
                with open(prompt_file, encoding="utf-8") as f:
                    payload = json.load(f)
                if not isinstance(payload, dict):
                    raise ValueError("prompt payload must be an object")
                prompt_name = str(payload.get("name") or payload.get("title") or f"Prompt {prompt_id_text}")
                if prefix_imported:
                    prompt_name = f"[Imported] {prompt_name}"
                get_by_name = getattr(prompts_db, "get_prompt_by_name", None)
                existing = get_by_name(prompt_name) if callable(get_by_name) else None
                if existing and conflict_resolution == ConflictResolution.SKIP:
                    status.skipped_items += 1
                    continue
                if existing and conflict_resolution == ConflictResolution.RENAME:
                    prompt_name = self._renamed_import_title(prompt_name)

                keywords = payload.get("keywords")
                if not isinstance(keywords, list):
                    keywords = []
                prompts_db.add_prompt(
                    name=prompt_name,
                    author=payload.get("author"),
                    details=payload.get("details") or payload.get("description"),
                    system_prompt=payload.get("system_prompt"),
                    user_prompt=payload.get("user_prompt"),
                    prompt_format=str(payload.get("prompt_format") or "legacy"),
                    prompt_schema_version=payload.get("prompt_schema_version"),
                    prompt_definition=payload.get("prompt_definition"),
                    keywords=[str(keyword) for keyword in keywords],
                    overwrite=False,
                )
                status.successful_items += 1
            except Exception as exc:
                status.failed_items += 1
                status.warnings.append(f"Error importing prompt {prompt_id_text}: {str(exc)}")

    def _import_evaluations(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        evaluation_ids: list[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob,
        manifest_path_index: ManifestImportPathIndex | None = None,
    ) -> None:
        """Import evaluation definitions and their archived runs."""
        evals_db = self._get_evaluations_db()
        if manifest_path_index is None:
            manifest_path_index = self._build_manifest_import_path_index(manifest)

        for evaluation_id in evaluation_ids:
            evaluation_id_text = str(evaluation_id)
            status.processed_items += 1
            if evals_db is None:
                status.failed_items += 1
                status.warnings.append("Evaluation import failed: evaluations database is unavailable.")
                continue
            eval_file, eval_path = self._resolve_manifest_import_file(
                extract_dir,
                manifest,
                ContentType.EVALUATION,
                evaluation_id_text,
                manifest_path_index,
            )
            if eval_file is None:
                status.failed_items += 1
                status.warnings.append(f"Unsafe evaluation file path: {eval_path}")
                continue
            if not eval_file.exists():
                status.failed_items += 1
                status.warnings.append(f"Evaluation file not found: {eval_path}")
                continue

            try:
                with open(eval_file, encoding="utf-8") as f:
                    payload = json.load(f)
                if not isinstance(payload, dict):
                    raise ValueError("evaluation payload must be an object")
                eval_name = str(payload.get("name") or f"Evaluation {evaluation_id_text}")
                if prefix_imported:
                    eval_name = f"[Imported] {eval_name}"
                existing = None
                try:
                    existing = evals_db.get_evaluation(str(payload.get("id") or evaluation_id_text))
                except Exception:
                    existing = None
                if existing and conflict_resolution == ConflictResolution.SKIP:
                    status.skipped_items += 1
                    continue
                eval_id_for_create = None if existing else str(payload.get("id") or evaluation_id_text)
                if existing and conflict_resolution == ConflictResolution.RENAME:
                    eval_name = self._renamed_import_title(eval_name)

                created_eval_id = evals_db.create_evaluation(
                    name=eval_name,
                    eval_type=str(payload.get("eval_type") or payload.get("type") or "custom"),
                    eval_spec=payload.get("eval_spec") if isinstance(payload.get("eval_spec"), dict) else {},
                    description=payload.get("description"),
                    dataset_id=payload.get("dataset_id"),
                    created_by=str(self.user_id),
                    metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
                    eval_id=eval_id_for_create,
                )
                for run in payload.get("runs") or []:
                    if not isinstance(run, dict):
                        continue
                    try:
                        run_id = evals_db.create_run(
                            created_eval_id,
                            target_model=run.get("target_model"),
                            config=run.get("config") if isinstance(run.get("config"), dict) else None,
                            webhook_url=run.get("webhook_url"),
                            run_id=run.get("id"),
                        )
                        with evals_db.get_connection() as conn:
                            conn.execute(
                                """
                                UPDATE evaluation_runs
                                   SET status = ?, progress = ?, results = ?, error_message = ?,
                                       started_at = ?, completed_at = ?, usage = ?
                                 WHERE id = ?
                                """,
                                (
                                    run.get("status") or "pending",
                                    json.dumps(run.get("progress")) if isinstance(run.get("progress"), dict) else run.get("progress"),
                                    json.dumps(run.get("results")) if isinstance(run.get("results"), (dict, list)) else run.get("results"),
                                    run.get("error_message"),
                                    run.get("started_at"),
                                    run.get("completed_at"),
                                    json.dumps(run.get("usage")) if isinstance(run.get("usage"), dict) else run.get("usage"),
                                    run_id,
                                ),
                            )
                            conn.commit()
                    except Exception as run_exc:
                        status.warnings.append(
                            f"Evaluation run restore failed for {evaluation_id_text}: {str(run_exc)}"
                        )
                status.successful_items += 1
            except Exception as exc:
                status.failed_items += 1
                status.warnings.append(f"Error importing evaluation {evaluation_id_text}: {str(exc)}")

    def _import_media_items(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        media_ids: list[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob,
        manifest_path_index: ManifestImportPathIndex | None = None,
        media_id_map: dict[str, int] | None = None,
    ) -> None:
        """Import Media DB rows, transcripts, chunks, and bundled stored artifacts."""
        media_db = self._get_media_db()
        if manifest_path_index is None:
            manifest_path_index = self._build_manifest_import_path_index(manifest)

        for media_id in media_ids:
            media_id_text = str(media_id)
            status.processed_items += 1
            if media_db is None:
                status.failed_items += 1
                status.warnings.append("Media import failed: media database is unavailable.")
                continue
            media_file, media_path = self._resolve_manifest_import_file(
                extract_dir,
                manifest,
                ContentType.MEDIA,
                media_id_text,
                manifest_path_index,
            )
            if media_file is None:
                status.failed_items += 1
                status.warnings.append(f"Unsafe media file path: {media_path}")
                continue
            if not media_file.exists():
                status.failed_items += 1
                status.warnings.append(f"Media file not found: {media_path}")
                continue

            try:
                with open(media_file, encoding="utf-8") as f:
                    payload = json.load(f)
                if not isinstance(payload, dict):
                    raise ValueError("media payload must be an object")
                media_title = str(payload.get("title") or f"Media {media_id_text}")
                if prefix_imported:
                    media_title = f"[Imported] {media_title}"
                get_by_title = getattr(media_db, "get_media_by_title", None)
                existing = get_by_title(media_title) if callable(get_by_title) else None
                if existing and conflict_resolution == ConflictResolution.SKIP:
                    status.skipped_items += 1
                    continue
                if existing and conflict_resolution == ConflictResolution.RENAME:
                    media_title = self._renamed_import_title(media_title)

                chunks = self._media_chunks_for_restore(payload)
                new_media_id, _media_uuid, _message = media_db.add_media_with_keywords(
                    url=payload.get("url"),
                    title=media_title,
                    media_type=payload.get("type") or payload.get("media_type") or "unknown",
                    content=payload.get("content") or "",
                    keywords=payload.get("keywords") if isinstance(payload.get("keywords"), list) else [],
                    prompt=payload.get("prompt"),
                    analysis_content=payload.get("analysis_content"),
                    safe_metadata=json.dumps(
                        {
                            "chatbook_import": {
                                "source_media_id": payload.get("id") or media_id_text,
                                "source_uuid": payload.get("uuid"),
                                "stored_artifacts": payload.get("stored_artifacts") or [],
                                "related_prompts": payload.get("related_prompts") or [],
                            }
                        },
                        ensure_ascii=False,
                    ),
                    source_hash=payload.get("source_hash"),
                    transcription_model=payload.get("transcription_model"),
                    author=payload.get("author"),
                    ingestion_date=payload.get("ingestion_date"),
                    overwrite=False,
                    chunks=chunks,
                    owner_user_id=self.user_id_int,
                )
                if not new_media_id:
                    status.failed_items += 1
                    continue
                if media_id_map is not None:
                    media_id_map[media_id_text] = int(new_media_id)
                    if payload.get("uuid"):
                        media_id_map[str(payload["uuid"])] = int(new_media_id)
                self._restore_media_transcripts(
                    media_db=media_db,
                    media_id=int(new_media_id),
                    payload=payload,
                    status=status,
                )
                restored_artifacts, pointer_only = self._restore_media_artifacts(
                    media_db=media_db,
                    extract_dir=extract_dir,
                    media_id=int(new_media_id),
                    stored_artifacts=[
                        item for item in payload.get("stored_artifacts") or [] if isinstance(item, dict)
                    ],
                    status=status,
                )
                if pointer_only:
                    status.warnings.append(
                        f"{pointer_only} stored media artifact pointer(s) for media {media_id_text} were restored as metadata only."
                    )
                if restored_artifacts:
                    status.warnings.append(
                        f"Restored {restored_artifacts} bundled stored media artifact(s) for media {media_id_text}."
                    )
                status.successful_items += 1
            except Exception as exc:
                status.failed_items += 1
                status.warnings.append(f"Error importing media {media_id_text}: {str(exc)}")

    def _import_embeddings(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        embedding_ids: list[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob,
        manifest_path_index: ManifestImportPathIndex | None = None,
        media_id_map: dict[str, int] | None = None,
    ) -> None:
        """Import Chroma collection embeddings and media vector blobs."""
        if manifest_path_index is None:
            manifest_path_index = self._build_manifest_import_path_index(manifest)
        chroma = None
        media_db = None

        for embedding_id in embedding_ids:
            embedding_id_text = str(embedding_id)
            status.processed_items += 1
            embedding_file, embedding_path = self._resolve_manifest_import_file(
                extract_dir,
                manifest,
                ContentType.EMBEDDING,
                embedding_id_text,
                manifest_path_index,
            )
            if embedding_file is None:
                status.failed_items += 1
                status.warnings.append(f"Unsafe embedding file path: {embedding_path}")
                continue
            if not embedding_file.exists():
                status.skipped_items += 1
                status.warnings.append(f"Embedding payload not bundled: {embedding_id_text}")
                continue
            try:
                with open(embedding_file, encoding="utf-8") as f:
                    payload = json.load(f)
                if not isinstance(payload, dict):
                    raise ValueError("embedding payload must be an object")
                if payload.get("bundled") is False:
                    status.skipped_items += 1
                    status.warnings.append(f"Embedding payload skipped because it was not bundled: {embedding_id_text}")
                    continue

                if str(payload.get("id") or embedding_id_text).startswith("media:"):
                    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
                    source_media_keys = [
                        str(source.get("media_id") or ""),
                        str(source.get("media_uuid") or ""),
                        str(embedding_id_text).split("media:", 1)[-1],
                    ]
                    target_media_id = None
                    for key in source_media_keys:
                        if key and media_id_map and key in media_id_map:
                            target_media_id = media_id_map[key]
                            break
                    if target_media_id is None:
                        status.skipped_items += 1
                        status.warnings.append(
                            f"Media embedding skipped because source media was not restored: {embedding_id_text}"
                        )
                        continue
                    media_db = media_db or self._get_media_db()
                    if media_db is None:
                        status.failed_items += 1
                        status.warnings.append("Media embedding import failed: media database is unavailable.")
                        continue
                    self._restore_media_vector_embedding(media_db, int(target_media_id), payload)
                    status.successful_items += 1
                    continue

                collection_name = str(payload.get("embedding_set_id") or "").strip()
                chunks = payload.get("chunks") if isinstance(payload.get("chunks"), list) else []
                if not collection_name or not chunks:
                    status.skipped_items += 1
                    status.warnings.append(f"Embedding collection payload had no restorable chunks: {embedding_id_text}")
                    continue
                chroma = chroma or self._get_chroma_manager()
                if chroma is None:
                    status.failed_items += 1
                    status.warnings.append("Embedding import failed: ChromaDB manager is unavailable.")
                    continue
                ids: list[str] = []
                texts: list[str] = []
                embeddings: list[list[float]] = []
                metadatas: list[dict[str, Any]] = []
                for chunk in chunks:
                    if not isinstance(chunk, dict) or not isinstance(chunk.get("embedding"), list):
                        continue
                    ids.append(str(chunk.get("id") or uuid4()))
                    texts.append(str(chunk.get("document") or ""))
                    embeddings.append(chunk["embedding"])
                    metadatas.append(chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {})
                if not ids:
                    status.skipped_items += 1
                    status.warnings.append(f"Embedding collection payload had no valid vectors: {embedding_id_text}")
                    continue
                importable_ids = self._filter_chroma_ids_for_conflict_resolution(
                    chroma=chroma,
                    collection_name=collection_name,
                    ids=ids,
                    conflict_resolution=conflict_resolution,
                    status=status,
                )
                if not importable_ids:
                    status.skipped_items += 1
                    continue
                filtered_indexes = [
                    index for index, item_id in enumerate(ids) if item_id in importable_ids
                ]
                chroma.store_in_chroma(
                    collection_name,
                    [texts[index] for index in filtered_indexes],
                    [embeddings[index] for index in filtered_indexes],
                    [ids[index] for index in filtered_indexes],
                    [metadatas[index] for index in filtered_indexes],
                )
                status.successful_items += 1
            except Exception as exc:
                status.failed_items += 1
                status.warnings.append(f"Error importing embedding {embedding_id_text}: {str(exc)}")

    def _import_generated_documents(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        document_ids: list[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob,
        manifest_path_index: ManifestImportPathIndex | None = None,
    ) -> None:
        """Import generic generated documents from a chatbook."""
        from ..Chat.document_generator import DocumentGeneratorService, DocumentType

        doc_service = DocumentGeneratorService(self.db, self.user_id)
        if manifest_path_index is None:
            manifest_path_index = self._build_manifest_import_path_index(manifest)

        for document_id in document_ids:
            document_id_text = str(document_id)
            status.processed_items += 1
            doc_file, doc_path = self._resolve_manifest_import_file(
                extract_dir,
                manifest,
                ContentType.GENERATED_DOCUMENT,
                document_id_text,
                manifest_path_index,
            )
            if doc_file is None:
                status.failed_items += 1
                status.warnings.append(f"Unsafe generated document file path: {doc_path}")
                continue
            if not doc_file.exists():
                status.failed_items += 1
                status.warnings.append(f"Generated document file not found: {doc_path}")
                continue
            try:
                with open(doc_file, encoding="utf-8") as f:
                    payload = json.load(f)
                if not isinstance(payload, dict):
                    raise ValueError("generated document payload must be an object")
                title = str(payload.get("title") or f"Generated document {document_id_text}")
                if prefix_imported:
                    title = f"[Imported] {title}"
                existing = None
                try:
                    existing = doc_service.get_document(str(payload.get("id") or document_id_text))
                except Exception:
                    existing = None
                if existing and conflict_resolution == ConflictResolution.SKIP:
                    status.skipped_items += 1
                    continue
                if existing and conflict_resolution == ConflictResolution.RENAME:
                    title = self._renamed_import_title(title)
                doc_type_raw = str(payload.get("document_type") or "summary")
                try:
                    doc_type = DocumentType(doc_type_raw)
                except Exception:
                    doc_type = DocumentType.SUMMARY
                new_document_id = doc_service._save_generated_document(
                    conversation_id=payload.get("conversation_id"),
                    document_type=doc_type,
                    title=title,
                    content=str(payload.get("content") or ""),
                    provider=str(payload.get("provider") or "imported"),
                    model=str(payload.get("model") or "imported"),
                    generation_time_ms=int(payload.get("generation_time_ms") or 0),
                    token_count=payload.get("token_count"),
                )
                if isinstance(payload.get("metadata"), dict) and payload["metadata"]:
                    with self.db.get_connection() as conn:
                        conn.execute(
                            "UPDATE generated_documents SET metadata = ? WHERE id = ?",
                            (json.dumps(payload["metadata"]), new_document_id),
                        )
                        conn.commit()
                status.successful_items += 1
            except Exception as exc:
                status.failed_items += 1
                status.warnings.append(f"Error importing generated document {document_id_text}: {str(exc)}")

    def _import_characters(
        self,
        extract_dir: Path,
        manifest: ChatbookManifest,
        character_ids: list[str],
        conflict_resolution: ConflictResolution,
        prefix_imported: bool,
        status: ImportJob,
        manifest_path_index: ManifestImportPathIndex | None = None,
        character_id_map: dict[str, int] | None = None
    ):
        """Import character cards from chatbook."""
        if manifest_path_index is None:
            manifest_path_index = self._build_manifest_import_path_index(manifest)

        for char_id in character_ids:
            status.processed_items += 1

            try:
                # Load character file
                char_file, char_path = self._resolve_manifest_import_file(
                    extract_dir,
                    manifest,
                    ContentType.CHARACTER,
                    char_id,
                    manifest_path_index,
                )
                if char_file is None:
                    status.failed_items += 1
                    status.warnings.append(f"Unsafe character file path: {char_path}")
                    continue
                if not char_file.exists():
                    status.failed_items += 1
                    status.warnings.append(f"Character file not found: {char_path}")
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
        status: ImportJob,
        manifest_path_index: ManifestImportPathIndex | None = None,
    ):
        """Import world books from chatbook."""
        # Import the world book service
        from ..Character_Chat.world_book_manager import WorldBookService
        wb_service = WorldBookService(self.db)
        if manifest_path_index is None:
            manifest_path_index = self._build_manifest_import_path_index(manifest)

        for wb_id in world_book_ids:
            status.processed_items += 1

            try:
                # Load world book file
                wb_file, wb_path = self._resolve_manifest_import_file(
                    extract_dir,
                    manifest,
                    ContentType.WORLD_BOOK,
                    wb_id,
                    manifest_path_index,
                )
                if wb_file is None:
                    status.failed_items += 1
                    status.warnings.append(f"Unsafe world book file path: {wb_path}")
                    continue
                if not wb_file.exists():
                    status.failed_items += 1
                    status.warnings.append(f"World book file not found: {wb_path}")
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
        status: ImportJob,
        manifest_path_index: ManifestImportPathIndex | None = None,
    ):
        """Import chat dictionaries from chatbook."""
        template_settings = self._resolve_template_settings(manifest)
        strict_dict_import = self._truthy_env("CHATBOOKS_IMPORT_DICT_STRICT", False)
        if manifest_path_index is None:
            manifest_path_index = self._build_manifest_import_path_index(manifest)

        # Import the dictionary service
        from ..Character_Chat.chat_dictionary import ChatDictionaryService
        dict_service = ChatDictionaryService(self.db)

        for dict_id in dictionary_ids:
            status.processed_items += 1

            try:
                # Load dictionary file
                dict_file, dict_path = self._resolve_manifest_import_file(
                    extract_dir,
                    manifest,
                    ContentType.DICTIONARY,
                    dict_id,
                    manifest_path_index,
                )
                if dict_file is None:
                    status.failed_items += 1
                    status.warnings.append(f"Unsafe dictionary file path: {dict_path}")
                    continue
                if not dict_file.exists():
                    status.failed_items += 1
                    status.warnings.append(f"Dictionary file not found: {dict_path}")
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

    def _cursor_count(self, cursor: Any) -> int:
        """Read a COUNT(*) result from supported cursor/list row shapes."""
        if hasattr(cursor, "fetchone"):
            row = cursor.fetchone()
        elif isinstance(cursor, list) and cursor:
            row = cursor[0]
        else:
            row = None
        if not row:
            return 0
        if isinstance(row, dict):
            return int(row.get("c", row.get("COUNT(1)", row.get("COUNT(*)", 0))) or 0)
        if hasattr(row, "keys"):
            try:
                return int(row["c"])
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                try:
                    return int(row["COUNT(1)"])
                except _CHATBOOK_NONCRITICAL_EXCEPTIONS:
                    return int(row[0])
        return int(row[0])

    def _count_operations_today_for_quota(self, operation_type: str) -> int:
        start = datetime.now(timezone.utc).date().isoformat()
        if operation_type == "export":
            cursor = self.db.execute_query(
                "SELECT COUNT(1) AS c FROM export_jobs WHERE user_id = ? AND created_at >= ?",
                (self.user_id, start),
            )
        else:
            cursor = self.db.execute_query(
                "SELECT COUNT(1) AS c FROM import_jobs WHERE user_id = ? AND created_at >= ?",
                (self.user_id, start),
            )
        return self._cursor_count(cursor)

    def _count_active_jobs_for_quota(self) -> int:
        export_cursor = self.db.execute_query(
            "SELECT COUNT(1) AS c FROM export_jobs WHERE user_id = ? AND status IN (?, ?)",
            (self.user_id, ExportStatus.PENDING.value, ExportStatus.IN_PROGRESS.value),
        )
        import_cursor = self.db.execute_query(
            "SELECT COUNT(1) AS c FROM import_jobs WHERE user_id = ? AND status IN (?, ?, ?)",
            (
                self.user_id,
                ImportStatus.PENDING.value,
                ImportStatus.VALIDATING.value,
                ImportStatus.IN_PROGRESS.value,
            ),
        )
        return self._cursor_count(export_cursor) + self._cursor_count(import_cursor)

    def _uses_postgres_backend(self) -> bool:
        backend_type = getattr(self.db, "backend_type", None)
        backend_value = getattr(backend_type, "value", backend_type)
        return str(backend_value).strip().lower() in {"postgres", "postgresql"}

    def _acquire_chatbook_quota_admission_lock(self) -> None:
        if not self._uses_postgres_backend():
            return
        self.db.execute_query(
            "SELECT pg_advisory_xact_lock(hashtext(?), hashtext(?))",
            ("chatbooks_quota", str(self.user_id)),
        )

    def _check_chatbook_job_admission(self, operation_type: str) -> None:
        quota_manager = QuotaManager(self.user_id, self.user_tier, db=self.db)
        if quota_manager._quotas_disabled:
            return

        if operation_type == "export":
            operation_limit = quota_manager.quotas["max_exports_per_day"]
            limit_message = f"Daily export limit ({operation_limit}) reached. Try again tomorrow."
            quota_type = "daily_export"
        else:
            operation_limit = quota_manager.quotas["max_imports_per_day"]
            limit_message = f"Daily import limit ({operation_limit}) reached. Try again tomorrow."
            quota_type = "daily_import"

        if operation_limit != UNLIMITED_QUOTA:
            operations_today = self._count_operations_today_for_quota(operation_type)
            if operations_today >= operation_limit:
                raise QuotaExceededError(
                    limit_message,
                    quota_type=quota_type,
                    limit=operation_limit,
                    current=operations_today,
                )

        concurrent_limit = quota_manager.quotas["max_concurrent_jobs"]
        if concurrent_limit != UNLIMITED_QUOTA:
            active_jobs = self._count_active_jobs_for_quota()
            if active_jobs >= concurrent_limit:
                raise QuotaExceededError(
                    f"Maximum concurrent jobs ({concurrent_limit}) reached. Wait for current jobs to complete.",
                    quota_type="concurrent_jobs",
                    limit=concurrent_limit,
                    current=active_jobs,
                )

    def _check_chatbook_job_admission_with_lock(self, operation_type: str) -> None:
        with self.db.transaction():
            self._acquire_chatbook_quota_admission_lock()
            self._check_chatbook_job_admission(operation_type)

    def _save_export_job_with_quota(self, job: ExportJob) -> None:
        with self.db.transaction():
            self._acquire_chatbook_quota_admission_lock()
            self._check_chatbook_job_admission("export")
            self._save_export_job(job, commit=False)

    def _save_import_job_with_quota(self, job: ImportJob) -> None:
        with self.db.transaction():
            self._acquire_chatbook_quota_admission_lock()
            self._check_chatbook_job_admission("import")
            self._save_import_job(job, commit=False)

    def _save_export_job(self, job: ExportJob, *, commit: bool = True):
        """Save export job to database.

        Note: Uses execute_query with commit=True by default, which handles its own transaction.
        Previous _with_transaction wrapper was removed because it created a separate
        connection that didn't share the transaction with execute_query's connection.
        """
        try:
            if job.status == ExportStatus.COMPLETED:
                job.progress_percentage = 100
                job.processed_items = job.total_items
            self.db.execute_query("""
                INSERT OR REPLACE INTO export_jobs (
                    job_id, user_id, status, chatbook_name, output_path,
                    created_at, started_at, completed_at, error_message,
                    progress_percentage, total_items, processed_items,
                    file_size_bytes, download_url, expires_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id, job.user_id, job.status.value, job.chatbook_name,
                job.output_path,
                self._serialize_job_timestamp(job.created_at),
                self._serialize_job_timestamp(job.started_at),
                self._serialize_job_timestamp(job.completed_at),
                job.error_message, job.progress_percentage, job.total_items,
                job.processed_items, job.file_size_bytes, job.download_url,
                self._serialize_job_timestamp(job.expires_at),
                json.dumps(job.metadata or {}, ensure_ascii=False)
            ), commit=commit)
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
            started_at = self._serialize_job_timestamp(datetime.now(timezone.utc))
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
            started_at = self._serialize_job_timestamp(datetime.now(timezone.utc))
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
                    'metadata': (row[15] if len(row) > 15 else (col13 if is_json_like else None)),
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

    def _save_import_job(self, job: ImportJob, *, commit: bool = True):
        """Save import job to database.

        Note: Uses execute_query with commit=True by default, which handles its own transaction.
        Previous _with_transaction wrapper was removed because it created a separate
        connection that didn't share the transaction with execute_query's connection.
        """
        try:
            if job.status == ImportStatus.COMPLETED:
                job.progress_percentage = 100
                job.processed_items = job.total_items
            self.db.execute_query("""
                INSERT OR REPLACE INTO import_jobs (
                    job_id, user_id, status, chatbook_path,
                    created_at, started_at, completed_at, error_message,
                    progress_percentage, total_items, processed_items,
                    successful_items, failed_items, skipped_items,
                    conflicts, warnings, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id, job.user_id, job.status.value, job.chatbook_path,
                self._serialize_job_timestamp(job.created_at),
                self._serialize_job_timestamp(job.started_at),
                self._serialize_job_timestamp(job.completed_at),
                job.error_message, job.progress_percentage, job.total_items,
                job.processed_items, job.successful_items, job.failed_items,
                job.skipped_items, json.dumps(job.conflicts), json.dumps(job.warnings),
                json.dumps(job.metadata or {}, ensure_ascii=False)
            ), commit=commit)
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
                    'warnings': row[15] if len(row) > 15 else '[]',
                    'metadata': row[16] if len(row) > 16 else '{}',
                }

            metadata = {}
            raw_metadata = row.get("metadata")
            if isinstance(raw_metadata, str):
                with contextlib.suppress(json.JSONDecodeError):
                    metadata = json.loads(raw_metadata)
            elif isinstance(raw_metadata, dict):
                metadata = raw_metadata

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
                warnings=json.loads(row['warnings']) if row['warnings'] else [],
                metadata=metadata,
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
        # Core backend: cancel queued or request cancel for processing in core Jobs
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
                    raise JobError(
                        "Saved export archive is outside managed storage; job history was preserved",
                        job_id=job_id,
                        job_type="export",
                    )
                if file_path.exists() and not file_path.is_file():
                    raise JobError(
                        "Saved export archive is not a regular file; job history was preserved",
                        job_id=job_id,
                        job_type="export",
                    )
                if file_path.is_file():
                    file_path.unlink()
            except JobError:
                raise
            except _CHATBOOK_NONCRITICAL_EXCEPTIONS as exc:
                logger.error(f"Failed to delete export file for job {job_id}: {exc}")
                raise JobError(
                    "Unable to delete saved export archive; job history was preserved",
                    job_id=job_id,
                    job_type="export",
                ) from exc

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

    def delete_finished_jobs(self, batch_size: int = 100) -> dict[str, int]:
        """Remove all terminal job records, including saved export archives."""
        batch_size = max(1, min(int(batch_size), 1000))
        export_jobs_removed = 0
        import_jobs_removed = 0

        for status in (
            ExportStatus.COMPLETED,
            ExportStatus.CANCELLED,
            ExportStatus.EXPIRED,
            ExportStatus.FAILED,
        ):
            while True:
                jobs = self.list_export_jobs(
                    status=status.value,
                    limit=batch_size,
                    offset=0,
                    raise_on_error=True,
                )
                if not jobs:
                    break
                removed_in_batch = sum(
                    1 for job in jobs if self.delete_export_job(job.job_id)
                )
                export_jobs_removed += removed_in_batch
                if removed_in_batch == 0:
                    raise JobError(
                        "Finished export job removal made no progress",
                        job_type="export",
                    )

        for status in (
            ImportStatus.COMPLETED,
            ImportStatus.CANCELLED,
            ImportStatus.FAILED,
        ):
            while True:
                jobs = self.list_import_jobs(
                    status=status.value,
                    limit=batch_size,
                    offset=0,
                    raise_on_error=True,
                )
                if not jobs:
                    break
                removed_in_batch = sum(
                    1 for job in jobs if self.delete_import_job(job.job_id)
                )
                import_jobs_removed += removed_in_batch
                if removed_in_batch == 0:
                    raise JobError(
                        "Finished import job removal made no progress",
                        job_type="import",
                    )

        return {
            "export_jobs_removed": export_jobs_removed,
            "import_jobs_removed": import_jobs_removed,
        }

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

    def _scope_db_ids_for_category(self, category: str) -> list[str]:
        """Return a best-effort ChaChaNotes-backed ID list for an account category."""
        list_ids = getattr(self.db, "list_chatbook_scope_ids", None)
        if not callable(list_ids):
            return []
        try:
            return [str(item) for item in list_ids(category) if item is not None]
        except _CHATBOOK_SCOPE_COUNT_EXCEPTIONS:
            return []

    def _list_all_prompt_ids(self) -> list[str]:
        if self._scope_existing_user_path(DatabasePaths.PROMPTS_SUBDIR, DatabasePaths.PROMPTS_DB_NAME) is None:
            return []
        prompts_db = self._get_prompts_db()
        if prompts_db is None:
            return []
        try:
            prompts, _page, _pages, _total = prompts_db.list_prompts(page=1, per_page=100000, sort_by="id", sort_order="asc")
            return [str(item["id"]) for item in prompts if item.get("id") is not None]
        except _CHATBOOK_SCOPE_COUNT_EXCEPTIONS:
            return []

    def _list_all_evaluation_ids(self) -> list[str]:
        if self._scope_existing_user_path(DatabasePaths.EVALUATIONS_SUBDIR, DatabasePaths.EVALUATIONS_DB_NAME) is None:
            return []
        evals_db = self._get_evaluations_db()
        if evals_db is None:
            return []
        try:
            total = max(1, int(evals_db.count_evaluations_filtered(created_by=self.user_id)))
            rows = evals_db.list_evaluations_filtered(limit=total, created_by=self.user_id)
            return [str(item["id"]) for item in rows if item.get("id") is not None]
        except _CHATBOOK_SCOPE_COUNT_EXCEPTIONS:
            return []

    def _list_all_explainer_session_ids(self) -> list[str]:
        if self._scope_existing_user_path(DatabasePaths.EXPLAINER_DB_NAME) is None:
            return []
        try:
            repo = self._get_explainer_repo()
            limit = 100
            rows, total = repo.list_session_summaries(owner_user_id=self.user_id, limit=limit, offset=0)
            all_rows = list(rows or [])
            offset = len(all_rows)
            while offset < int(total or 0):
                page, _total = repo.list_session_summaries(
                    owner_user_id=self.user_id, limit=limit, offset=offset
                )
                if not page:
                    break
                all_rows.extend(page)
                offset += len(page)
            return [str(getattr(item, "id", "")) for item in all_rows if getattr(item, "id", None)]
        except _CHATBOOK_SCOPE_COUNT_EXCEPTIONS:
            return []

    def _list_all_media_ids(self) -> list[str]:
        if self._scope_existing_user_path(DatabasePaths.MEDIA_DB_NAME) is None:
            return []
        media_db = self._get_media_db()
        if media_db is None:
            return []
        list_ids = getattr(media_db, "list_chatbook_scope_ids", None)
        if not callable(list_ids):
            return []
        try:
            return [str(item) for item in list_ids("media_records") if item is not None]
        except _CHATBOOK_SCOPE_COUNT_EXCEPTIONS:
            return []

    def _expand_full_account_content_selections(self) -> dict[ContentType, list[str]]:
        """Expand full-account export to every content collector this service owns."""
        selections = {
            ContentType.CONVERSATION: self._scope_db_ids_for_category("conversations"),
            ContentType.NOTE: self._scope_db_ids_for_category("notes"),
            ContentType.CHARACTER: self._scope_db_ids_for_category("characters"),
            ContentType.WORLD_BOOK: self._scope_db_ids_for_category("world_books"),
            ContentType.DICTIONARY: self._scope_db_ids_for_category("dictionaries"),
            ContentType.PROMPT: self._list_all_prompt_ids(),
            ContentType.EVALUATION: self._list_all_evaluation_ids(),
            ContentType.MEDIA: self._list_all_media_ids(),
            ContentType.GENERATED_DOCUMENT: self._scope_db_ids_for_category("generated_documents"),
            ContentType.EXPLAINER_SESSION: self._list_all_explainer_session_ids(),
            ContentType.EMBEDDING: [],
        }
        chroma_path = self._scope_existing_user_path(DatabasePaths.CHROMA_SUBDIR)
        if chroma_path is None:
            selections.pop(ContentType.EMBEDDING)
        return selections

    def _scope_existing_user_path(self, *parts: str) -> Path | None:
        """Return an existing user-owned path without creating optional stores."""
        user_id_value = self.user_id_int if self.user_id_int is not None else self.user_id
        try:
            path = DatabasePaths.resolve_user_base_directory(user_id_value).joinpath(*parts)
            return path if path.exists() else None
        except _CHATBOOK_SCOPE_COUNT_EXCEPTIONS:
            return None

    @staticmethod
    def _coerce_scope_count(value: Any) -> int:
        """Normalize DB-layer scope counts into non-negative integers."""
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def _scope_db_count_for_category(self, category: str) -> int:
        """Return a best-effort ChaChaNotes-backed count for an account category."""
        count_category = getattr(self.db, "count_chatbook_scope_category", None)
        if not callable(count_category):
            return 0
        try:
            return self._coerce_scope_count(count_category(category))
        except _CHATBOOK_SCOPE_COUNT_EXCEPTIONS:
            return 0

    def _scope_media_count_for_category(self, category: str) -> int:
        """Return a best-effort Media DB-backed count for an account category."""
        if self._scope_existing_user_path(DatabasePaths.MEDIA_DB_NAME) is None:
            return 0
        media_db = self._get_media_db()
        if media_db is None:
            return 0
        count_category = getattr(media_db, "count_chatbook_scope_category", None)
        if not callable(count_category):
            return 0
        try:
            return self._coerce_scope_count(count_category(category))
        except _CHATBOOK_SCOPE_COUNT_EXCEPTIONS:
            return 0

    def _scope_count_for_category(self, category: str) -> int:
        """Best-effort category counts for the full-account export preview."""
        if category in {"account_profile", "account_settings"}:
            return 1
        if category in {"conversations", "notes", "characters", "world_books", "dictionaries"}:
            return self._scope_db_count_for_category(category)
        if category == "prompts":
            if self._scope_existing_user_path(DatabasePaths.PROMPTS_SUBDIR, DatabasePaths.PROMPTS_DB_NAME) is None:
                return 0
            prompts_db = self._get_prompts_db()
            if prompts_db is None:
                return 0
            try:
                return max(0, int(prompts_db.list_prompts(page=1, per_page=1)[3]))
            except _CHATBOOK_SCOPE_COUNT_EXCEPTIONS:
                return 0
        if category == "evaluations":
            if self._scope_existing_user_path(DatabasePaths.EVALUATIONS_SUBDIR, DatabasePaths.EVALUATIONS_DB_NAME) is None:
                return 0
            evals_db = self._get_evaluations_db()
            if evals_db is None:
                return 0
            try:
                return max(0, int(evals_db.count_evaluations_filtered(created_by=self.user_id)))
            except _CHATBOOK_SCOPE_COUNT_EXCEPTIONS:
                return 0
        if category == "generated_documents":
            return self._scope_db_count_for_category(category)
        if category == "explainer_sessions":
            if self._scope_existing_user_path(DatabasePaths.EXPLAINER_DB_NAME) is None:
                return 0
            try:
                return max(0, int(self._get_explainer_repo().list_session_summaries(owner_user_id=self.user_id, limit=1)[1]))
            except _CHATBOOK_SCOPE_COUNT_EXCEPTIONS:
                return 0
        if category in {
            "media_records",
            "media_transcripts",
            "media_chunks",
            "media_stored_artifacts",
            "media_pointers",
        }:
            return self._scope_media_count_for_category(category)
        if category == "embeddings":
            chroma_path = self._scope_existing_user_path(DatabasePaths.CHROMA_SUBDIR)
            if chroma_path is None:
                return 0
            try:
                if not any(chroma_path.iterdir()):
                    return 0
            except _CHATBOOK_SCOPE_COUNT_EXCEPTIONS:
                return 0
            chroma = self._get_chroma_manager()
            if chroma is None:
                return 0
            try:
                return sum(max(0, int(collection.count())) for collection in chroma.list_collections())
            except _CHATBOOK_SCOPE_COUNT_EXCEPTIONS:
                return 0
        return 0

    def get_full_account_export_scope(self) -> dict[str, Any]:
        """Return a redacted full-account Chatbook export scope summary."""
        categories = [
            {
                "category": row.category,
                "label": row.label,
                "count": self._scope_count_for_category(row.category),
                "restore_status": row.restore_status,
                "sensitivity": row.sensitivity,
                "warning": row.warning,
            }
            for row in ACCOUNT_DATA_INVENTORY
        ]
        return {
            "mode": FULL_ACCOUNT_EXPORT_MODE,
            "categories": categories,
            "total_items": sum(item["count"] for item in categories),
            "pointer_only_count": sum(item["count"] for item in categories if item["restore_status"] == "pointer_only"),
            "sensitive_category_count": sum(
                1 for item in categories if item["sensitivity"] in {"sensitive", "secret"}
            ),
            "warning_count": sum(1 for item in categories if item["warning"]),
            "estimated_size_bytes": None,
        }

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

    # Removed legacy JobQueueShim handlers; Chatbooks uses core Jobs.

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
        if manifest.total_explainer_sessions > 0:
            content.append(f"- **Explainer Sessions:** {manifest.total_explainer_sessions}\n")

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
            if manifest.total_explainer_sessions > 0:
                f.write(f"- **Explainer Sessions:** {manifest.total_explainer_sessions}\n")

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
