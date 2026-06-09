# voice_manager.py
# Description: Voice management service for handling custom voice uploads and processing
#
# Imports
import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

#
# Third-party Imports
import aiofiles
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from tldw_Server_API.app.core.AuthNZ.exceptions import QuotaExceededError, StorageError
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import (
    FILE_CATEGORY_VOICE_CLONE,
    SOURCE_FEATURE_VOICE_STUDIO,
)
from tldw_Server_API.app.core.config import settings

#
# Local Imports
from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    DatabasePaths,
    _normalize_user_db_base_dir,
)
from tldw_Server_API.app.core.DB_Management.Voice_Registry_DB import VoiceRegistryDB
from tldw_Server_API.app.core.Storage.generated_file_helpers import AUDIO_MIME_TYPES
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat
from tldw_Server_API.app.services.storage_quota_service import get_storage_service

from .tts_exceptions import TTSError
from .utils import parse_bool

#
#######################################################################################################################
#
# Voice Management Classes and Service

# Provider requirements for voice samples
PROVIDER_REQUIREMENTS = {
    "vibevoice": {
        "formats": [".wav", ".mp3", ".flac", ".ogg"],
        "max_size_mb": 50,
        "duration": {"min": 0.1, "max": 600},  # 0.1s to 10 minutes
        "sample_rate": 22050,
        "convert_to": "wav"
    },
    "higgs": {
        "formats": [".wav", ".mp3"],
        "max_size_mb": 10,
        "duration": {"min": 3, "max": 10},
        "sample_rate": 16000,
        "convert_to": "wav"
    },
    "chatterbox": {
        "formats": [".wav", ".mp3"],
        "max_size_mb": 20,
        "duration": {"min": 5, "max": 20},
        "sample_rate": 24000,
        "convert_to": "wav"
    },
    "elevenlabs": {
        "formats": [".wav", ".mp3"],
        "max_size_mb": 10,
        "duration": {"min": 1, "max": 30},
        "sample_rate": 44100,
        "convert_to": "mp3"
    },
    "neutts": {
        "formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"],
        "max_size_mb": 20,
        "duration": {"min": 3, "max": 15},
        "sample_rate": 16000,
        "convert_to": "wav"
    },
    "pocket_tts": {
        "formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a"],
        "max_size_mb": 20,
        "duration": {"min": 1, "max": 60},
        "sample_rate": 24000,
        "convert_to": "wav"
    },
    "pocket_tts_cpp": {
        "formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a"],
        "max_size_mb": 20,
        "duration": {"min": 1, "max": 60},
        "sample_rate": 24000,
        "convert_to": "wav"
    },
    "qwen3_tts": {
        "formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"],
        "max_size_mb": 50,
        # Qwen3-TTS highlights rapid voice cloning from ~3s references in the README.
        "duration": {"min": 3, "max": 30},
        "sample_rate": 24000,
        "convert_to": "wav"
    },
    "omnivoice": {
        "formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"],
        "max_size_mb": 50,
        "duration": {"min": 3, "max": 30},
        "sample_rate": 24000,
        "convert_to": "wav",
    }
}

# Rate limiting configuration
VOICE_RATE_LIMITS = {
    "upload_per_hour": 5,
    "total_storage_mb": 500,
    "concurrent_processing": 2,
    "max_voices_per_user": 50
}
VOICE_REGISTRY_COMPAT_MODE_REMOVAL_DATE = "2026-12-31"

DEFAULT_NEUTTS_VOICE_ID = "default"
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_NEUTTS_VOICE_PATH = (
    _REPO_ROOT / "Helper_Scripts" / "Audio" / "Sample_Voices" / "Sample_Voice_1.wav"
)
DEFAULT_NEUTTS_VOICE_TEXT_PATH = DEFAULT_NEUTTS_VOICE_PATH.with_suffix(".txt")

_VOICE_IO_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    UnicodeError,
)

_VOICE_METADATA_EXCEPTIONS = _VOICE_IO_EXCEPTIONS + (
    json.JSONDecodeError,
    KeyError,
    RuntimeError,
    AttributeError,
)

_VOICE_NONCRITICAL_EXCEPTIONS = _VOICE_METADATA_EXCEPTIONS + (
    StorageError,
    QuotaExceededError,
    asyncio.TimeoutError,
)


class VoiceUploadRequest(BaseModel):
    """Request model for voice upload"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    provider: str = Field(default="vibevoice")
    reference_text: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional transcript of the reference audio for cloning providers.",
    )

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        # Remove any path traversal attempts
        if any(char in v for char in ['/', '\\', '..', '~']):
            raise ValueError("Voice name contains invalid characters")
        return v.strip()


class VoiceInfo(BaseModel):
    """Information about a voice sample"""
    voice_id: str
    name: str
    description: Optional[str] = None
    file_path: str
    format: str
    duration: float
    sample_rate: Optional[int] = None
    size_bytes: int
    provider: str
    created_at: datetime
    file_hash: str


class VoiceUploadResponse(BaseModel):
    """Response model for voice upload"""
    voice_id: str
    name: str
    file_path: str
    duration: float
    format: str
    provider_compatible: bool
    warnings: list[str] = []
    info: str = ""


class VoiceReferenceMetadata(BaseModel):
    """Stored metadata and provider artifacts for a voice reference."""
    voice_id: str
    reference_text: Optional[str] = None
    voice_clone_prompt_b64: Optional[str] = None
    voice_clone_prompt_format: Optional[str] = None
    provider_artifacts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def touch(self) -> None:
        self.updated_at = datetime.utcnow()


class VoiceEncodeResult(BaseModel):
    """Result of encoding provider-specific artifacts for a stored voice."""
    voice_id: str
    provider: str
    cached: bool = False
    ref_codes_len: Optional[int] = None
    reference_text: Optional[str] = None


class VoiceProcessingError(TTSError):
    """Base exception for voice processing errors"""
    pass


class VoiceFormatError(VoiceProcessingError):
    """Invalid audio format error"""
    pass


class VoiceDurationError(VoiceProcessingError):
    """Voice duration outside acceptable range"""
    pass


class VoiceQuotaExceededError(VoiceProcessingError):
    """User exceeded voice storage quota"""
    pass


class VoiceFileValidator:
    """Validates voice file uploads"""

    ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.opus'}
    ALLOWED_MIME_TYPES = {
        'audio/wav', 'audio/x-wav', 'audio/wave',
        'audio/mpeg', 'audio/mp3',
        'audio/flac', 'audio/x-flac',
        'audio/ogg', 'application/ogg',
        'audio/mp4', 'audio/x-m4a',
        'audio/opus'
    }
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB default

    @staticmethod
    def validate_filename(filename: str) -> tuple[bool, str]:
        """Validate and sanitize filename"""
        if not filename:
            return False, "No filename provided"

        # Get extension
        ext = Path(filename).suffix.lower()
        if ext not in VoiceFileValidator.ALLOWED_EXTENSIONS:
            return False, f"File type {ext} not allowed. Allowed types: {', '.join(VoiceFileValidator.ALLOWED_EXTENSIONS)}"

        # Sanitize filename
        safe_name = "".join(c for c in filename if c.isalnum() or c in '._- ')
        safe_name = safe_name.replace(' ', '_')

        return True, safe_name

    @staticmethod
    def validate_file_size(size_bytes: int, provider: str = "vibevoice") -> tuple[bool, str]:
        """Validate file size for provider"""
        max_size = PROVIDER_REQUIREMENTS.get(provider, {}).get("max_size_mb", 50) * 1024 * 1024

        if size_bytes > max_size:
            return False, f"File size {size_bytes / 1024 / 1024:.1f}MB exceeds limit of {max_size / 1024 / 1024}MB"

        if size_bytes == 0:
            return False, "File is empty"

        return True, ""

    @staticmethod
    def sanitize_path(base_path: Path, filename: str) -> Path:
        """Ensure path is safe and within base directory"""
        # Remove any path components from filename
        safe_name = Path(filename).name
        full_path = base_path / safe_name

        # Resolve to absolute path and check it's within base
        try:
            full_path = full_path.resolve()
            base_path = base_path.resolve()
            full_path.relative_to(base_path)
        except (ValueError, RuntimeError) as e:
            raise VoiceProcessingError(f"Invalid file path: {e}") from e

        return full_path


class VoiceRegistry:
    """In-memory registry for voice samples"""

    def __init__(self):
        self.user_voices: dict[int, dict[str, VoiceInfo]] = {}
        self._lock = asyncio.Lock()

    async def register_voice(self, user_id: int, voice_info: VoiceInfo):
        """Register a voice in the runtime registry"""
        async with self._lock:
            if user_id not in self.user_voices:
                self.user_voices[user_id] = {}

            self.user_voices[user_id][voice_info.voice_id] = voice_info
            logger.info(f"Registered voice {voice_info.name} ({voice_info.voice_id}) for user {user_id}")

    async def get_voice(self, user_id: int, voice_id: str) -> Optional[VoiceInfo]:
        """Get voice info from registry"""
        async with self._lock:
            return self.user_voices.get(user_id, {}).get(voice_id)

    async def list_voices(self, user_id: int) -> list[VoiceInfo]:
        """List all voices for a user"""
        async with self._lock:
            return list(self.user_voices.get(user_id, {}).values())

    async def remove_voice(self, user_id: int, voice_id: str) -> bool:
        """Remove voice from registry"""
        async with self._lock:
            if user_id in self.user_voices and voice_id in self.user_voices[user_id]:
                del self.user_voices[user_id][voice_id]
                logger.info(f"Removed voice {voice_id} for user {user_id}")
                return True
            return False

    async def clear_user_voices(self, user_id: int):
        """Clear all voices for a user"""
        async with self._lock:
            if user_id in self.user_voices:
                del self.user_voices[user_id]
                logger.info(f"Cleared all voices for user {user_id}")


class VoiceManager:
    """Main voice management service"""

    def __init__(self):
        self.registry = VoiceRegistry()
        self.processing_queue = asyncio.Queue()
        self.cleanup_interval = 3600  # 1 hour
        self.user_upload_counts: dict[int, list[datetime]] = {}
        self._registry_store_cache: dict[str, VoiceRegistryDB] = {}
        self._voice_registry_disable_warning_emitted = False
        self._processing_tasks: dict[str, asyncio.Task] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_stop_event: Optional[asyncio.Event] = None
        # Tracks filesystem snapshot for fast cross-process registry invalidation.
        # Format: {user_id: (voice_file_count, newest_mtime_ns)}
        self._registry_snapshots: dict[int, tuple[int, int]] = {}

    def _invalidate_registry_snapshot(self, user_id: int) -> None:
        """Mark a user's cached registry snapshot stale."""
        self._registry_snapshots.pop(user_id, None)

    def _get_processed_snapshot(self, user_id: int) -> tuple[int, int]:
        """Return a lightweight snapshot of processed voice files for a user."""
        voices_path = self.get_user_voices_path(user_id)
        processed_path = voices_path / "processed"
        if not processed_path.exists():
            return (0, 0)

        count = 0
        newest_mtime_ns = 0
        for voice_file in processed_path.iterdir():
            if not voice_file.is_file():
                continue
            if voice_file.suffix.lower() not in VoiceFileValidator.ALLOWED_EXTENSIONS:
                continue
            try:
                stat = voice_file.stat()
            except _VOICE_IO_EXCEPTIONS:
                continue
            count += 1
            newest_mtime_ns = max(newest_mtime_ns, int(stat.st_mtime_ns))

        return (count, newest_mtime_ns)

    def _is_persistent_registry_enabled(self) -> bool:
        """Return True when persistent voice registry storage is enabled."""
        raw_override = os.getenv("TTS_VOICE_REGISTRY_ENABLED")
        if raw_override is None:
            raw_override = settings.get("TTS_VOICE_REGISTRY_ENABLED")
        enabled = parse_bool(raw_override, default=True)
        if not enabled and not self._voice_registry_disable_warning_emitted:
            logger.warning(
                "Persistent voice registry disabled via TTS_VOICE_REGISTRY_ENABLED. "
                "Compatibility mode is deprecated and scheduled for removal after {}.",
                VOICE_REGISTRY_COMPAT_MODE_REMOVAL_DATE,
            )
            self._voice_registry_disable_warning_emitted = True
        return enabled

    def get_user_voice_registry_db_path(self, user_id: int) -> Path:
        """Get the SQLite path for persistent voice registry records."""
        voices_path = self.get_user_voices_path(user_id)
        return voices_path / DatabasePaths.VOICE_REGISTRY_DB_NAME

    def _get_registry_store(self, user_id: int) -> VoiceRegistryDB:
        if not self._is_persistent_registry_enabled():
            raise RuntimeError("Persistent voice registry disabled")
        db_path = str(self.get_user_voice_registry_db_path(user_id).resolve())
        store = self._registry_store_cache.get(db_path)
        if store is None:
            store = VoiceRegistryDB(db_path)
            self._registry_store_cache[db_path] = store
        return store

    def _get_registry_store_safe(self, user_id: int) -> Optional[VoiceRegistryDB]:
        """Return persistent registry store, or None when unavailable."""
        if not self._is_persistent_registry_enabled():
            return None
        try:
            return self._get_registry_store(user_id)
        except _VOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(
                "Persistent voice registry unavailable for user {}: {}",
                user_id,
                e,
            )
            return None

    @staticmethod
    def _to_registry_record(voice: VoiceInfo) -> dict[str, Any]:
        return {
            "voice_id": voice.voice_id,
            "name": voice.name,
            "description": voice.description,
            "file_path": voice.file_path,
            "format": voice.format,
            "duration": voice.duration,
            "sample_rate": voice.sample_rate,
            "size_bytes": voice.size_bytes,
            "provider": voice.provider,
            "created_at": voice.created_at.isoformat(),
            "file_hash": voice.file_hash,
        }

    @staticmethod
    def _from_registry_record(record: dict[str, Any]) -> VoiceInfo:
        created_at_raw = record.get("created_at")
        try:
            created_at = datetime.fromisoformat(str(created_at_raw))
        except (ValueError, TypeError):
            created_at = datetime.utcnow()
        # Defense-in-depth: reject file_path values that contain path-traversal
        # segments or are absolute paths.  Downstream consumers already have
        # resolve()+relative_to() guards, but we reject bad data at the
        # deserialization boundary as well.
        raw_file_path = str(record.get("file_path") or "")
        if os.path.isabs(raw_file_path) or ".." in raw_file_path.replace("\\", "/").split("/"):
            logger.warning("Ignoring persisted voice with unsafe file_path: {}", raw_file_path)
            raw_file_path = ""

        return VoiceInfo(
            voice_id=str(record.get("voice_id") or ""),
            name=str(record.get("name") or ""),
            description=record.get("description"),
            file_path=raw_file_path,
            format=str(record.get("format") or "wav"),
            duration=float(record.get("duration") or 0.0),
            sample_rate=record.get("sample_rate"),
            size_bytes=int(record.get("size_bytes") or 0),
            provider=str(record.get("provider") or "vibevoice"),
            created_at=created_at,
            file_hash=str(record.get("file_hash") or ""),
        )

    async def _list_persisted_voices(self, user_id: int) -> list[VoiceInfo]:
        store = self._get_registry_store_safe(user_id)
        if store is None:
            return []
        try:
            rows = await asyncio.to_thread(store.list_voices, user_id)
            return [self._from_registry_record(row) for row in rows]
        except Exception as e:
            logger.warning("Failed to list persisted voices for user {}: {}", user_id, e)
            return []

    async def _get_persisted_voice(self, user_id: int, voice_id: str) -> Optional[VoiceInfo]:
        store = self._get_registry_store_safe(user_id)
        if store is None:
            return None
        try:
            row = await asyncio.to_thread(store.get_voice, user_id, voice_id)
            if row is None:
                return None
            return self._from_registry_record(row)
        except Exception as e:
            logger.warning(
                "Failed to read persisted voice {} for user {}: {}",
                voice_id,
                user_id,
                e,
            )
            return None

    async def _upsert_persisted_voice(self, user_id: int, voice: VoiceInfo) -> None:
        store = self._get_registry_store_safe(user_id)
        if store is None:
            return
        try:
            await asyncio.to_thread(store.upsert_voice, user_id, self._to_registry_record(voice))
        except Exception as e:
            logger.warning("Failed to upsert persisted voice {} for user {}: {}", voice.voice_id, user_id, e)

    async def _replace_persisted_voices(self, user_id: int, voices: list[VoiceInfo]) -> None:
        store = self._get_registry_store_safe(user_id)
        if store is None:
            return
        records = [self._to_registry_record(voice) for voice in voices]
        try:
            await asyncio.to_thread(store.replace_user_voices, user_id, records)
        except Exception as e:
            logger.warning("Failed to replace persisted voices for user {}: {}", user_id, e)

    async def _delete_persisted_voice(self, user_id: int, voice_id: str) -> bool:
        store = self._get_registry_store_safe(user_id)
        if store is None:
            return False
        try:
            return await asyncio.to_thread(store.delete_voice, user_id, voice_id)
        except Exception as e:
            logger.warning(
                "Failed to delete persisted voice {} for user {}: {}",
                voice_id,
                user_id,
                e,
            )
            return False

    async def _refresh_runtime_registry(self, user_id: int, voices: list[VoiceInfo]) -> None:
        await self.registry.clear_user_voices(user_id)
        for voice in voices:
            await self.registry.register_voice(user_id, voice)

    def _merge_with_persisted_metadata(
        self,
        scanned_voices: list[VoiceInfo],
        persisted_voices: list[VoiceInfo],
    ) -> list[VoiceInfo]:
        persisted_by_id = {voice.voice_id: voice for voice in persisted_voices}
        merged: list[VoiceInfo] = []
        for scanned in scanned_voices:
            persisted = persisted_by_id.get(scanned.voice_id)
            if persisted is None:
                merged.append(scanned)
                continue
            merged.append(
                scanned.model_copy(
                    update={
                        "name": persisted.name or scanned.name,
                        "description": (
                            persisted.description
                            if persisted.description is not None
                            else scanned.description
                        ),
                        "provider": persisted.provider or scanned.provider,
                        "sample_rate": (
                            persisted.sample_rate
                            if persisted.sample_rate is not None
                            else scanned.sample_rate
                        ),
                        "created_at": persisted.created_at,
                        "file_hash": persisted.file_hash or scanned.file_hash,
                    }
                )
            )
        return merged

    async def _sync_registry_from_filesystem(self, user_id: int, *, force: bool = False) -> list[VoiceInfo]:
        """Refresh runtime and persistent registries when filesystem state changes."""
        current_snapshot = self._get_processed_snapshot(user_id)
        cached_snapshot = self._registry_snapshots.get(user_id)
        if not self._is_persistent_registry_enabled():
            current_registry = await self.registry.list_voices(user_id)
            if not force and cached_snapshot == current_snapshot and current_registry:
                return current_registry
            voices = await self._scan_user_voices(user_id)
            await self._refresh_runtime_registry(user_id, voices)
            self._registry_snapshots[user_id] = self._get_processed_snapshot(user_id)
            return voices

        persisted_voices = await self._list_persisted_voices(user_id)
        should_reconcile = force or cached_snapshot != current_snapshot or not persisted_voices

        if should_reconcile:
            scanned_voices = await self._scan_user_voices(user_id)
            voices = self._merge_with_persisted_metadata(scanned_voices, persisted_voices)
            await self._replace_persisted_voices(user_id, voices)
        else:
            stale_ids = [
                voice.voice_id
                for voice in persisted_voices
                if not self._voice_file_exists(user_id, voice)
            ]
            if stale_ids:
                for stale_id in stale_ids:
                    await self._delete_persisted_voice(user_id, stale_id)
                stale_set = set(stale_ids)
                voices = [
                    voice for voice in persisted_voices if voice.voice_id not in stale_set
                ]
            else:
                voices = persisted_voices

        await self._refresh_runtime_registry(user_id, voices)
        self._registry_snapshots[user_id] = self._get_processed_snapshot(user_id)
        return voices

    def _voice_file_exists(self, user_id: int, voice: VoiceInfo) -> bool:
        """Return True if the registry entry points to an existing in-bounds file."""
        voices_path = self.get_user_voices_path(user_id).resolve()
        try:
            candidate = (voices_path / voice.file_path).resolve()
            candidate.relative_to(voices_path)
        except _VOICE_IO_EXCEPTIONS:
            return False
        return candidate.exists()

    async def _unregister_voice_clone_generated_files(
        self,
        *,
        user_id: int,
        storage_path: str,
    ) -> None:
        """
        Best-effort unregister generated_files entries for a voice clone artifact.

        This keeps quota/accounting records aligned when deleting voice references.
        """
        normalized_target = str(storage_path or "").replace("\\", "/").strip()
        if not normalized_target:
            return

        target_name = Path(normalized_target).name
        try:
            storage_service = await get_storage_service()
            files_repo = await storage_service.get_generated_files_repo()
            rows, _total = await files_repo.list_files(
                user_id=user_id,
                file_category=FILE_CATEGORY_VOICE_CLONE,
                source_feature=SOURCE_FEATURE_VOICE_STUDIO,
                offset=0,
                limit=500,
            )
            for row in rows:
                file_id = row.get("id")
                if not isinstance(file_id, int):
                    continue
                row_storage_path = str(row.get("storage_path") or "").replace("\\", "/").strip()
                if (
                    row_storage_path == normalized_target
                    or row_storage_path.endswith(f"/{normalized_target}")
                    or Path(row_storage_path).name == target_name
                ):
                    await storage_service.unregister_generated_file(file_id, hard_delete=True)
        except _VOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(
                "Failed to unregister generated voice-clone file for user {} path {}: {}",
                user_id,
                normalized_target,
                e,
            )

    def get_user_voices_path(self, user_id: int) -> Path:
        """Get the voices directory path for a user.

        Uses DatabasePaths to resolve `<USER_DB_BASE_DIR>/<user_id>/voices`.
        """
        # Normalize user_id to a safe, canonical string to avoid unsafe path components.
        # This will raise ValueError if user_id cannot be interpreted as an integer,
        # preventing unexpected directory traversal via crafted IDs.
        safe_user_id_str = str(int(user_id))

        sample_root = DEFAULT_NEUTTS_VOICE_PATH.parent.resolve()
        env_user_db_base = os.getenv("USER_DB_BASE_DIR")
        settings_user_db_base = settings.get("USER_DB_BASE_DIR")
        test_context = bool(os.getenv("PYTEST_CURRENT_TEST")) or is_test_mode()
        if test_context and env_user_db_base:
            user_db_base = env_user_db_base
        else:
            user_db_base = settings_user_db_base or env_user_db_base
        if user_db_base:
            base_dir = _normalize_user_db_base_dir(Path(user_db_base))
        else:
            if test_context:
                base_dir = (Path.cwd() / "Databases" / "user_databases").resolve()
            else:
                base_dir = (_REPO_ROOT / "Databases" / "user_databases").resolve()
        candidate_dir = (base_dir / safe_user_id_str / DatabasePaths.VOICES_SUBDIR).resolve()
        try:
            candidate_dir.relative_to(sample_root)
        except ValueError:
            return DatabasePaths.get_user_voices_dir(safe_user_id_str)
        fallback_base = (_REPO_ROOT / "Databases" / "user_databases").resolve()
        logger.warning(
            "Voices directory resolved under Sample_Voices; falling back to {}",
            fallback_base,
        )
        return DatabasePaths.get_user_voices_dir(
            safe_user_id_str,
            base_dir_override=fallback_base,
        )

    def get_user_voice_metadata_path(self, user_id: int, voice_id: str) -> Path:
        """Get the metadata path for a stored voice reference."""
        voices_path = self.get_user_voices_path(user_id)
        metadata_dir = voices_path / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize voice_id to prevent path traversal
        safe_voice_id = Path(voice_id).name
        if not safe_voice_id or safe_voice_id != voice_id:
            raise VoiceProcessingError(f"Invalid voice_id: {voice_id}")

        full_path = (metadata_dir / f"{safe_voice_id}.json").resolve()
        # Verify path stays within metadata_dir
        try:
            full_path.relative_to(metadata_dir.resolve())
        except ValueError as e:
            raise VoiceProcessingError(f"Invalid metadata path: {e}") from e

        return full_path

    async def load_reference_metadata(
        self, user_id: int, voice_id: str
    ) -> Optional[VoiceReferenceMetadata]:
        """Load stored reference metadata if it exists."""
        path = self.get_user_voice_metadata_path(user_id, voice_id)
        if not path.exists():
            return None
        try:
            async with aiofiles.open(path) as f:
                raw = await f.read()
            data = json.loads(raw)
            return VoiceReferenceMetadata(**data)
        except _VOICE_METADATA_EXCEPTIONS as e:
            logger.warning(f"Failed to read voice metadata {path}: {e}")
            return None

    async def save_reference_metadata(
        self, user_id: int, metadata: VoiceReferenceMetadata
    ) -> None:
        """Persist reference metadata for a voice."""
        metadata.touch()
        path = self.get_user_voice_metadata_path(user_id, metadata.voice_id)
        try:
            payload = model_dump_compat(metadata)
            async with aiofiles.open(path, "w") as f:
                await f.write(json.dumps(payload, ensure_ascii=True, indent=2, default=str))
        except _VOICE_METADATA_EXCEPTIONS as e:
            logger.warning(f"Failed to write voice metadata {path}: {e}")

    async def ensure_default_voice(self, user_id: int) -> Optional[VoiceInfo]:
        """Ensure the bundled default NeuTTS voice exists for a user."""
        existing = await self.registry.get_voice(user_id, DEFAULT_NEUTTS_VOICE_ID)
        if existing is None:
            existing = await self._get_persisted_voice(user_id, DEFAULT_NEUTTS_VOICE_ID)
            if existing is not None:
                await self.registry.register_voice(user_id, existing)
        if existing and self._voice_file_exists(user_id, existing):
            return existing
        if existing:
            await self.registry.remove_voice(user_id, DEFAULT_NEUTTS_VOICE_ID)
            await self._delete_persisted_voice(user_id, DEFAULT_NEUTTS_VOICE_ID)

        voices_path = self.get_user_voices_path(user_id)
        processed_path = voices_path / "processed"
        for ext in sorted(VoiceFileValidator.ALLOWED_EXTENSIONS):
            candidate = processed_path / f"{DEFAULT_NEUTTS_VOICE_ID}{ext}"
            if candidate.exists():
                voice_info = await self._build_voice_info_from_file(
                    voice_id=DEFAULT_NEUTTS_VOICE_ID,
                    name="Default",
                    description="Bundled NeuTTS default voice",
                    provider="neutts",
                    voices_path=voices_path,
                    audio_path=candidate,
                )
                await self.registry.register_voice(user_id, voice_info)
                await self._upsert_persisted_voice(user_id, voice_info)
                self._invalidate_registry_snapshot(user_id)
                return voice_info

        if not DEFAULT_NEUTTS_VOICE_PATH.exists():
            logger.debug(
                f"Default NeuTTS voice not found at {DEFAULT_NEUTTS_VOICE_PATH}"
            )
            return None

        try:
            processed_file = await self._process_for_provider(
                DEFAULT_NEUTTS_VOICE_PATH,
                processed_path / f"{DEFAULT_NEUTTS_VOICE_ID}.wav",
                "neutts",
            )
            voice_info = await self._build_voice_info_from_file(
                voice_id=DEFAULT_NEUTTS_VOICE_ID,
                name="Default",
                description="Bundled NeuTTS default voice",
                provider="neutts",
                voices_path=voices_path,
                audio_path=processed_file,
            )
            await self.registry.register_voice(user_id, voice_info)
            await self._upsert_persisted_voice(user_id, voice_info)
            self._invalidate_registry_snapshot(user_id)

            reference_text = None
            if DEFAULT_NEUTTS_VOICE_TEXT_PATH.exists():
                try:
                    async with aiofiles.open(DEFAULT_NEUTTS_VOICE_TEXT_PATH) as f:
                        reference_text = (await f.read()).strip() or None
                except _VOICE_IO_EXCEPTIONS as e:
                    logger.warning(f"Failed to read default voice reference text: {e}")

            if reference_text:
                metadata = VoiceReferenceMetadata(
                    voice_id=DEFAULT_NEUTTS_VOICE_ID,
                    reference_text=reference_text,
                )
                await self.save_reference_metadata(user_id, metadata)
                try:
                    await self.encode_voice_reference(
                        user_id=user_id,
                        voice_id=DEFAULT_NEUTTS_VOICE_ID,
                        provider="neutts",
                        reference_text=reference_text,
                        force=False,
                    )
                except VoiceProcessingError as e:
                    logger.warning(f"Default NeuTTS auto-encode failed: {e}")

            return voice_info
        except _VOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Failed to register default NeuTTS voice: {e}")
            return None

    async def _build_voice_info_from_file(
        self,
        *,
        voice_id: str,
        name: str,
        description: Optional[str],
        provider: str,
        voices_path: Path,
        audio_path: Path,
    ) -> VoiceInfo:
        duration = await self._get_audio_duration(audio_path)
        provider_reqs = PROVIDER_REQUIREMENTS.get(provider, {})
        try:
            with open(audio_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
        except _VOICE_IO_EXCEPTIONS:
            file_hash = ""
        return VoiceInfo(
            voice_id=voice_id,
            name=name,
            description=description,
            file_path=str(audio_path.relative_to(voices_path)),
            format=audio_path.suffix[1:],
            duration=duration,
            sample_rate=provider_reqs.get("sample_rate"),
            size_bytes=audio_path.stat().st_size,
            provider=provider,
            created_at=datetime.utcnow(),
            file_hash=file_hash,
        )

    async def _get_voice_info(self, user_id: int, voice_id: str) -> Optional[VoiceInfo]:
        return await self.get_voice(user_id, voice_id)

    async def get_voice(self, user_id: int, voice_id: str, *, refresh: bool = False) -> Optional[VoiceInfo]:
        """Get a voice with automatic filesystem-backed registry synchronization."""
        if voice_id == DEFAULT_NEUTTS_VOICE_ID:
            await self.ensure_default_voice(user_id)
        await self._sync_registry_from_filesystem(user_id, force=refresh)
        voice = await self.registry.get_voice(user_id, voice_id)
        if voice and not self._voice_file_exists(user_id, voice):
            await self._sync_registry_from_filesystem(user_id, force=True)
            return await self.registry.get_voice(user_id, voice_id)
        return voice

    async def _get_voice_audio_path(self, user_id: int, voice_id: str) -> Path:
        voice_info = await self._get_voice_info(user_id, voice_id)
        if not voice_info:
            raise VoiceProcessingError(f"Voice not found: {voice_id}")
        voices_path = self.get_user_voices_path(user_id)
        audio_path = (voices_path / voice_info.file_path).resolve()
        try:
            audio_path.relative_to(voices_path.resolve())
        except Exception as e:
            raise VoiceProcessingError(f"Invalid voice path for {voice_id}: {e}") from e
        if not audio_path.exists():
            raise VoiceProcessingError(f"Voice file missing for {voice_id}")
        return audio_path

    async def load_voice_reference_audio(self, user_id: int, voice_id: str) -> bytes:
        """Load stored voice reference audio bytes."""
        if voice_id == DEFAULT_NEUTTS_VOICE_ID:
            await self.ensure_default_voice(user_id)
        audio_path = await self._get_voice_audio_path(user_id, voice_id)
        try:
            async with aiofiles.open(audio_path, "rb") as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Failed to read voice audio for {voice_id}: {e}")
            raise VoiceProcessingError(f"Failed to read voice audio for {voice_id}") from e

    async def check_rate_limits(self, user_id: int) -> tuple[bool, str]:
        """Check if user is within rate limits"""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)

        # Clean old entries and count recent uploads
        if user_id not in self.user_upload_counts:
            self.user_upload_counts[user_id] = []

        self.user_upload_counts[user_id] = [
            dt for dt in self.user_upload_counts[user_id]
            if dt > hour_ago
        ]
        memory_recent_upload_count = len(self.user_upload_counts[user_id])
        fs_recent_upload_count = 0
        # Cross-instance consistency: also inspect upload artifacts on disk.
        voices_path = self.get_user_voices_path(user_id)
        uploads_dir = voices_path / "uploads"
        if uploads_dir.exists():
            for upload_file in uploads_dir.glob("*"):
                if not upload_file.is_file():
                    continue
                try:
                    mtime = datetime.utcfromtimestamp(upload_file.stat().st_mtime)
                except _VOICE_IO_EXCEPTIONS:
                    continue
                if mtime > hour_ago:
                    fs_recent_upload_count += 1

        recent_upload_count = max(memory_recent_upload_count, fs_recent_upload_count)

        if recent_upload_count >= VOICE_RATE_LIMITS["upload_per_hour"]:
            return False, f"Rate limit exceeded: {VOICE_RATE_LIMITS['upload_per_hour']} uploads per hour"

        # Check total storage
        provider_cache_root = (voices_path / "providers").resolve()
        total_size = 0
        for file_path in voices_path.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                resolved = file_path.resolve()
                resolved.relative_to(provider_cache_root)
                continue
            except ValueError:
                pass
            except _VOICE_IO_EXCEPTIONS:
                continue
            try:
                total_size += file_path.stat().st_size
            except _VOICE_IO_EXCEPTIONS:
                continue

        max_storage = VOICE_RATE_LIMITS["total_storage_mb"] * 1024 * 1024
        if total_size > max_storage:
            return False, f"Storage quota exceeded: {total_size / 1024 / 1024:.1f}MB / {VOICE_RATE_LIMITS['total_storage_mb']}MB"

        # Check voice count from filesystem (cross-instance safe).
        voice_count, _ = self._get_processed_snapshot(user_id)
        if voice_count >= VOICE_RATE_LIMITS["max_voices_per_user"]:
            return False, f"Maximum voice limit reached: {VOICE_RATE_LIMITS['max_voices_per_user']} voices"

        return True, ""

    async def upload_voice(
        self,
        user_id: int,
        file_content: bytes,
        filename: str,
        request: VoiceUploadRequest
    ) -> VoiceUploadResponse:
        """Process a voice upload"""

        # Check rate limits
        can_upload, error_msg = await self.check_rate_limits(user_id)
        if not can_upload:
            raise VoiceQuotaExceededError(error_msg)

        # Validate filename
        is_valid, safe_filename = VoiceFileValidator.validate_filename(filename)
        if not is_valid:
            raise VoiceFormatError(safe_filename)

        # Validate file size
        is_valid, error_msg = VoiceFileValidator.validate_file_size(len(file_content), request.provider)
        if not is_valid:
            raise VoiceProcessingError(error_msg)

        # Generate unique ID
        voice_id = str(uuid.uuid4())

        # Save original file
        voices_path = self.get_user_voices_path(user_id)
        uploads_dir = voices_path / "uploads"
        upload_path = VoiceFileValidator.sanitize_path(
            uploads_dir,
            f"{voice_id}_{safe_filename}"
        )

        try:
            async with aiofiles.open(upload_path, 'wb') as f:
                await f.write(file_content)

            logger.info(f"Saved uploaded voice file: {upload_path}")

            # Calculate file hash
            file_hash = hashlib.sha256(file_content).hexdigest()

            # Get audio duration (simplified - in production use ffprobe or similar)
            duration = await self._get_audio_duration(upload_path)

            # Validate duration for provider
            provider_reqs = PROVIDER_REQUIREMENTS.get(request.provider, {})
            min_duration = provider_reqs.get("duration", {}).get("min", 0)
            max_duration = provider_reqs.get("duration", {}).get("max", float('inf'))

            warnings = []
            if duration < min_duration:
                warnings.append(f"Audio duration {duration:.1f}s is less than recommended {min_duration}s for {request.provider}")
            elif duration > max_duration:
                warnings.append(f"Audio duration {duration:.1f}s exceeds maximum {max_duration}s for {request.provider}")

            # Optional strict enforcement for production deployments: when
            # TTS_VOICE_STRICT_DURATION is truthy, reject uploads that fall
            # outside the recommended duration range instead of only warning.
            if warnings and parse_bool(os.getenv("TTS_VOICE_STRICT_DURATION"), default=False):
                raise VoiceDurationError(
                    f"Voice sample duration {duration:.1f}s is outside the recommended "
                    f"range [{min_duration}, {max_duration}] seconds for provider '{request.provider}'"
                )

            # Process for provider (convert format if needed)
            processed_path = await self._process_for_provider(
                upload_path,
                voices_path / "processed" / f"{voice_id}.wav",
                request.provider
            )

            # Register voice clone with storage tracking (quota + generated files)
            try:
                storage_service = await get_storage_service()
                storage_path = str(processed_path.relative_to(voices_path))
                mime_type = AUDIO_MIME_TYPES.get(processed_path.suffix.lstrip(".").lower(), "application/octet-stream")
                tags = [f"voice:{request.name}"]
                if request.provider:
                    tags.append(f"provider:{request.provider}")
                await storage_service.register_generated_file(
                    user_id=user_id,
                    filename=processed_path.name,
                    storage_path=storage_path,
                    file_category=FILE_CATEGORY_VOICE_CLONE,
                    source_feature=SOURCE_FEATURE_VOICE_STUDIO,
                    file_size_bytes=processed_path.stat().st_size,
                    original_filename=filename,
                    mime_type=mime_type,
                    source_ref=f"provider:{request.provider}" if request.provider else None,
                    tags=tags,
                    check_quota=True,
                )
            except QuotaExceededError as exc:
                raise VoiceQuotaExceededError(str(exc)) from exc
            except StorageError as exc:
                raise VoiceProcessingError(f"Failed to register voice with storage: {exc}") from exc

            # Create voice info
            voice_info = VoiceInfo(
                voice_id=voice_id,
                name=request.name,
                description=request.description,
                file_path=str(processed_path.relative_to(voices_path)),
                format=processed_path.suffix[1:],
                duration=duration,
                sample_rate=provider_reqs.get("sample_rate"),
                size_bytes=processed_path.stat().st_size,
                provider=request.provider,
                created_at=datetime.utcnow(),
                file_hash=file_hash
            )

            # Register voice
            await self.registry.register_voice(user_id, voice_info)
            await self._upsert_persisted_voice(user_id, voice_info)
            self._invalidate_registry_snapshot(user_id)

            # Store optional reference metadata
            if request.reference_text:
                metadata = VoiceReferenceMetadata(
                    voice_id=voice_id,
                    reference_text=request.reference_text.strip() or None,
                )
                await self.save_reference_metadata(user_id, metadata)
                if (request.provider or "").strip().lower() == "neutts":
                    try:
                        await self.encode_voice_reference(
                            user_id=user_id,
                            voice_id=voice_id,
                            provider="neutts",
                            reference_text=metadata.reference_text,
                            force=False,
                        )
                    except VoiceProcessingError as e:
                        warnings.append(f"NeuTTS auto-encode failed: {e}")

            # Update upload count
            self.user_upload_counts[user_id].append(datetime.utcnow())

            return VoiceUploadResponse(
                voice_id=voice_id,
                name=request.name,
                file_path=str(processed_path),
                duration=duration,
                format=voice_info.format,
                provider_compatible=len(warnings) == 0,
                warnings=warnings,
                info=f"Voice '{request.name}' uploaded successfully for {request.provider}"
            )

        except VoiceProcessingError:
            # Clean up on error but preserve the specific voice processing
            # exception type (e.g., VoiceDurationError) so API layers can
            # distinguish between validation and generic failures.
            if upload_path.exists():
                upload_path.unlink()
            if "processed_path" in locals() and processed_path.exists():
                processed_path.unlink()
            raise
        except _VOICE_NONCRITICAL_EXCEPTIONS as e:
            # Clean up on error
            if upload_path.exists():
                upload_path.unlink()
            if "processed_path" in locals() and processed_path.exists():
                processed_path.unlink()
            logger.error(f"Failed to upload voice: {e}")
            raise VoiceProcessingError(f"Failed to process voice upload: {str(e)}") from e

    async def encode_voice_reference(
        self,
        user_id: int,
        voice_id: str,
        provider: str,
        reference_text: Optional[str] = None,
        force: bool = False,
    ) -> VoiceEncodeResult:
        """Encode provider-specific artifacts for a stored voice reference."""
        provider_key = (provider or "").strip().lower()
        if not provider_key:
            raise VoiceProcessingError("Provider is required for voice encoding")

        metadata = await self.load_reference_metadata(user_id, voice_id)
        if metadata is None:
            metadata = VoiceReferenceMetadata(voice_id=voice_id)

        if reference_text:
            metadata.reference_text = reference_text.strip() or None

        existing = metadata.provider_artifacts.get(provider_key)
        if existing and not force:
            ref_codes = existing.get("ref_codes")
            return VoiceEncodeResult(
                voice_id=voice_id,
                provider=provider_key,
                cached=True,
                ref_codes_len=len(ref_codes) if isinstance(ref_codes, list) else None,
                reference_text=existing.get("reference_text") or metadata.reference_text,
            )

        audio_path = await self._get_voice_audio_path(user_id, voice_id)

        if provider_key == "neutts":
            ref_text = metadata.reference_text
            if not ref_text:
                raise VoiceProcessingError("reference_text is required to encode NeuTTS ref_codes")
            ref_codes = await self._encode_neutts_reference(audio_path)
            metadata.provider_artifacts[provider_key] = {
                "ref_codes": ref_codes,
                "reference_text": ref_text,
            }
            await self.save_reference_metadata(user_id, metadata)
            return VoiceEncodeResult(
                voice_id=voice_id,
                provider=provider_key,
                cached=False,
                ref_codes_len=len(ref_codes),
                reference_text=ref_text,
            )

        if provider_key == "omnivoice":
            ref_text = metadata.reference_text
            if not ref_text:
                raise VoiceProcessingError("reference_text is required to encode OmniVoice references")
            metadata.provider_artifacts[provider_key] = {
                "reference_text": ref_text,
            }
            await self.save_reference_metadata(user_id, metadata)
            return VoiceEncodeResult(
                voice_id=voice_id,
                provider=provider_key,
                cached=False,
                ref_codes_len=None,
                reference_text=ref_text,
            )

        raise VoiceProcessingError(f"Provider not supported for encoding: {provider_key}")

    async def _encode_neutts_reference(self, audio_path: Path) -> list[int]:
        """Encode NeuTTS reference codes from a stored audio file."""
        try:
            from tldw_Server_API.app.core.TTS.adapters.neutts_adapter import NeuTTSAdapter
            from tldw_Server_API.app.core.TTS.tts_config import get_tts_config_manager

            manager = get_tts_config_manager()
            provider_cfg = manager.get_provider_config("neutts")
            cfg = model_dump_compat(provider_cfg) if provider_cfg is not None else {}
            adapter = NeuTTSAdapter(config=cfg)
            await adapter.ensure_initialized()
            codes = adapter._engine.encode_reference(str(audio_path))  # type: ignore[attr-defined]
            if hasattr(codes, "tolist"):
                codes = codes.tolist()
            if not isinstance(codes, (list, tuple)):
                raise VoiceProcessingError("NeuTTS encoder returned invalid ref_codes")
            return [int(x) for x in codes]
        except VoiceProcessingError:
            raise
        except Exception as e:
            logger.error(f"NeuTTS reference encoding failed: {e}")
            raise VoiceProcessingError(f"Failed to encode NeuTTS reference: {e}") from e

    async def _get_audio_duration(self, file_path: Path) -> float:
        """Get audio file duration using ffprobe (non-blocking)."""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(file_path)
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
                with contextlib.suppress(_VOICE_NONCRITICAL_EXCEPTIONS):
                    await proc.communicate()
                logger.error(f"ffprobe timed out for {file_path}")
                return 0.0

            if proc.returncode == 0 and stdout:
                try:
                    return float(stdout.decode().strip())
                except (UnicodeDecodeError, ValueError) as e:
                    logger.error(f"Error parsing ffprobe output for {file_path}: {e}")
                    return 0.0

            err_msg = (stderr or b"").decode(errors="ignore") if stderr is not None else ""
            logger.warning(f"Could not determine audio duration for {file_path}: {err_msg}")
            return 0.0

        except FileNotFoundError as e:
            logger.error(f"ffprobe not found while getting audio duration: {e}")
            return 0.0
        except _VOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error getting audio duration for {file_path}: {e}")
            return 0.0

    async def _get_audio_sample_rate(self, file_path: Path) -> Optional[int]:
        """Get audio sample rate using ffprobe (non-blocking)."""
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(file_path)
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
                with contextlib.suppress(_VOICE_NONCRITICAL_EXCEPTIONS):
                    await proc.communicate()
                logger.error(f"ffprobe timed out while getting audio sample rate for {file_path}")
                return None

            if proc.returncode == 0 and stdout:
                try:
                    return int(stdout.decode().strip())
                except (UnicodeDecodeError, ValueError) as e:
                    logger.error(f"Error parsing ffprobe sample rate output for {file_path}: {e}")
                    return None

            err_msg = (stderr or b"").decode(errors="ignore") if stderr is not None else ""
            logger.warning(f"Could not determine audio sample rate for {file_path}: {err_msg}")
            return None

        except FileNotFoundError as e:
            logger.error(f"ffprobe not found while getting audio sample rate: {e}")
            return None
        except _VOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error getting audio sample rate for {file_path}: {e}")
            return None

    async def _process_for_provider(self, input_path: Path, output_path: Path, provider: str) -> Path:
        """Process audio file for specific provider requirements"""
        provider_reqs = PROVIDER_REQUIREMENTS.get(provider, {})
        target_format = provider_reqs.get("convert_to", "wav")
        target_sr = provider_reqs.get("sample_rate") or 22050

        # If already in correct format and sample rate, just copy.
        if input_path.suffix[1:].lower() == str(target_format).lower():
            current_sr = await self._get_audio_sample_rate(input_path)
            if current_sr is not None and int(current_sr) != int(target_sr):
                logger.info(
                    f"Resampling {input_path} for {provider}: {current_sr}Hz -> {target_sr}Hz"
                )
            elif current_sr is not None:
                shutil.copy2(input_path, output_path)
                return output_path
            else:
                logger.info(f"Sample rate unknown for {input_path}; normalizing with ffmpeg")

        # Convert using ffmpeg
        try:
            # Ensure output has correct extension
            output_path = output_path.with_suffix(f".{target_format}")

            cmd = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-ar', str(target_sr),  # Sample rate
                '-ac', '1',  # Mono
                '-c:a', 'pcm_s16le' if target_format == 'wav' else 'libmp3lame',
                str(output_path)
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            except asyncio.TimeoutError:
                proc.kill()
                with contextlib.suppress(_VOICE_NONCRITICAL_EXCEPTIONS):
                    await proc.communicate()
                logger.error(f"FFmpeg conversion timed out for {input_path}")
                shutil.copy2(input_path, output_path)
                return output_path

            if proc.returncode != 0:
                err_msg = (stderr or b"").decode(errors="ignore") if stderr is not None else ""
                logger.error(f"FFmpeg conversion failed for {input_path}: {err_msg}")
                # Fall back to copying original
                shutil.copy2(input_path, output_path)
            else:
                logger.info(f"Converted audio to {target_format} at {target_sr}Hz")

            return output_path
        except _VOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Audio conversion failed: {e}")
            # Fall back to copying original
            shutil.copy2(input_path, output_path)
            return output_path

    async def list_user_voices(self, user_id: int, *, refresh: bool = False) -> list[VoiceInfo]:
        """List all voices for a user"""
        await self.ensure_default_voice(user_id)
        return await self._sync_registry_from_filesystem(user_id, force=refresh)

    async def _scan_user_voices(self, user_id: int) -> list[VoiceInfo]:
        """Scan filesystem for user's voices"""
        voices = []
        voices_path = self.get_user_voices_path(user_id)
        processed_path = voices_path / "processed"

        if processed_path.exists():
            for voice_file in processed_path.glob("*"):
                if voice_file.is_file() and voice_file.suffix in VoiceFileValidator.ALLOWED_EXTENSIONS:
                    try:
                        # Extract provider and voice ID from filename. By default, files are
                        # named `<voice_id>.<ext>`. For future multi-provider layouts we
                        # also support an optional `provider__voice_id.ext` pattern.
                        stem = voice_file.stem
                        provider_name = "vibevoice"
                        voice_id = stem
                        if "__" in stem:
                            maybe_provider, maybe_id = stem.split("__", 1)
                            if maybe_provider:
                                provider_name = maybe_provider.lower()
                            if maybe_id:
                                voice_id = maybe_id
                        elif stem == DEFAULT_NEUTTS_VOICE_ID:
                            provider_name = "neutts"

                        # Defense-in-depth: reject voice_id values derived from
                        # filenames that contain path separators or traversal sequences.
                        if not voice_id or "/" in voice_id or "\\" in voice_id or ".." in voice_id:
                            logger.warning("Skipping voice file with unsafe stem: {}", voice_file.name)
                            continue

                        # Get file info
                        stat = voice_file.stat()
                        duration = await self._get_audio_duration(voice_file)

                        # Create voice info
                        voice_info = VoiceInfo(
                            voice_id=voice_id,
                            name="Default" if voice_id == DEFAULT_NEUTTS_VOICE_ID else voice_id,
                            description="Bundled NeuTTS default voice" if voice_id == DEFAULT_NEUTTS_VOICE_ID else None,
                            file_path=str(voice_file.relative_to(voices_path)),
                            format=voice_file.suffix[1:],
                            duration=duration,
                            size_bytes=stat.st_size,
                            provider=provider_name,
                            created_at=datetime.fromtimestamp(stat.st_ctime),
                            file_hash=""  # Would need to calculate
                        )

                        voices.append(voice_info)

                    except _VOICE_NONCRITICAL_EXCEPTIONS as e:
                        logger.error(f"Error scanning voice file {voice_file}: {e}")

        return voices

    async def delete_voice(self, user_id: int, voice_id: str) -> bool:
        """Delete a voice"""
        # Get voice info
        voice_info = await self.get_voice(user_id, voice_id)
        if not voice_info:
            return False

        # Delete files
        voices_path = self.get_user_voices_path(user_id)
        storage_relative_path = str(voice_info.file_path)

        # Best-effort storage accounting cleanup for generated voice-clone records.
        await self._unregister_voice_clone_generated_files(
            user_id=user_id,
            storage_path=storage_relative_path,
        )

        # Delete processed file
        try:
            voices_base = voices_path.resolve()
            processed_file = (voices_path / voice_info.file_path).resolve()
            processed_file.relative_to(voices_base)
        except (ValueError, RuntimeError) as e:
            logger.error(
                f"Refusing to delete voice {voice_id} for user {user_id}: invalid path "
                f"{voice_info.file_path} ({e})"
            )
            return False
        if processed_file.exists():
            processed_file.unlink()

        # Delete original upload if exists
        uploads_dir = (voices_path / "uploads").resolve()
        safe_prefix = f"{voice_id}_"
        if uploads_dir.exists():
            for upload_file in uploads_dir.iterdir():
                if not upload_file.is_file() or not upload_file.name.startswith(safe_prefix):
                    continue
                try:
                    upload_file.resolve().relative_to(uploads_dir)
                    upload_file.unlink()
                except (ValueError, RuntimeError):
                    logger.warning(f"Skipping invalid upload file path: {upload_file}")

        # Delete reference metadata json if present
        try:
            metadata_path = self.get_user_voice_metadata_path(user_id, voice_id)
            if metadata_path.exists():
                metadata_path.unlink()
        except _VOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Failed to delete metadata for voice {voice_id}: {e}")

        # Remove from registry
        await self.registry.remove_voice(user_id, voice_id)
        await self._delete_persisted_voice(user_id, voice_id)
        self._invalidate_registry_snapshot(user_id)

        logger.info(f"Deleted voice {voice_id} for user {user_id}")
        return True

    async def cleanup_temp_files(self):
        """Clean up old temporary files"""
        try:
            # Clean all user temp directories
            base_path = DatabasePaths.get_user_db_base_dir()
            if base_path.exists():
                for user_dir in base_path.iterdir():
                    if user_dir.is_dir():
                        # Defense-in-depth: ensure user_dir resolves inside
                        # base_path (guards against symlinks escaping).
                        try:
                            user_dir.resolve().relative_to(base_path.resolve())
                        except ValueError:
                            continue
                        temp_dir = user_dir / "voices" / "temp"
                        if temp_dir.exists():
                            # Remove files older than 1 hour
                            cutoff_time = datetime.utcnow().timestamp() - 3600
                            resolved_temp_dir = temp_dir.resolve()
                            for temp_file in temp_dir.iterdir():
                                if temp_file.is_file() and temp_file.stat().st_mtime < cutoff_time:
                                    try:
                                        temp_file.resolve().relative_to(resolved_temp_dir)
                                    except (ValueError, RuntimeError):
                                        continue
                                    temp_file.unlink()
                                    logger.debug(f"Cleaned up temp file: {temp_file}")

        except _VOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error during temp file cleanup: {e}")

    async def start_background_tasks(self):
        """Start background processing tasks"""
        if self._cleanup_task and not self._cleanup_task.done():
            return
        self._cleanup_stop_event = asyncio.Event()
        self._cleanup_task = asyncio.create_task(self._cleanup_worker(self._cleanup_stop_event))
        logger.info("Voice manager background tasks started")

    async def stop_background_tasks(self):
        """Stop background processing tasks."""
        if self._cleanup_task is None:
            return

        stop_event = self._cleanup_stop_event
        if stop_event is not None:
            stop_event.set()

        cleanup_task = self._cleanup_task
        try:
            await asyncio.wait_for(cleanup_task, timeout=5.0)
        except asyncio.TimeoutError:
            cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task
        finally:
            self._cleanup_task = None
            self._cleanup_stop_event = None
            logger.info("Voice manager background tasks stopped")

    async def _cleanup_worker(self, stop_event: asyncio.Event):
        """Background worker for cleanup"""
        while not stop_event.is_set():
            try:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.cleanup_interval)
                    break
                except asyncio.TimeoutError:
                    pass
                await self.cleanup_temp_files()
            except _VOICE_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Cleanup worker error: {e}")


# Global instance
_voice_manager: Optional[VoiceManager] = None


def get_voice_manager() -> VoiceManager:
    """Get or create the global voice manager instance"""
    global _voice_manager
    if _voice_manager is None:
        _voice_manager = VoiceManager()
    return _voice_manager


async def init_voice_manager():
    """Initialize the voice manager and start background tasks"""
    manager = get_voice_manager()
    await manager.start_background_tasks()
    logger.info("Voice manager initialized")


async def shutdown_voice_manager() -> bool:
    """Shutdown the global voice manager background tasks."""
    global _voice_manager
    if _voice_manager is None:
        return False
    await _voice_manager.stop_background_tasks()
    return True
