# Audio_Transcription_Lib.py
#########################################
# Transcription Library
# This library is used to perform transcription of audio files.
# Currently, uses faster_whisper for transcription.
#
####################
# Function List
#
# 1. convert_to_wav(video_file_path, offset=0, overwrite=False)
# 2. speech_to_text(audio_file_path, selected_source_lang='en', whisper_model='small.en', vad_filter=False)
#
####################
#
# Import necessary libraries to run solo for testing
import asyncio
import gc
import hashlib
import importlib
import inspect
import json
import multiprocessing
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Sequence
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

#
# DEBUG Imports
#from memory_profiler import profile

# Import diarization module (optional dependency)
try:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Diarization_Lib import (
        DiarizationError,
        DiarizationService,
        load_diarization_config,
    )
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    DiarizationService = None
    DiarizationError = Exception  # Fallback to base Exception
    def load_diarization_config():
        return {}  # type: ignore[assignment]
#
# Import Local
import contextlib

from tldw_Server_API.app.core.config import (
    get_stt_config,
    load_and_log_configs,
    loaded_config_data,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.exceptions import (
    CancelCheckError,
    STTTranscriptionError,
    TranscriptionCancelled,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import resolve_safe_local_path
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram, timeit
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.Utils.Utils import logging, sanitize_filename

_AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    subprocess.SubprocessError,
    DiarizationError,
    CancelCheckError,
    TranscriptionCancelled,
)

torch: Any | None = None
_TORCH_IMPORT_ERROR: Exception | None = None
_TORCH_IMPORT_ATTEMPTED: bool = False
_ORIGINAL_WHISPER_MODEL: Any | None = None
_ORIGINAL_WHISPER_MODEL_IMPORT_ERROR: Exception | None = None
_ORIGINAL_WHISPER_MODEL_IMPORT_ATTEMPTED: bool = False
_QWEN2AUDIO_CLASSES: tuple[Any, Any] | None = None
_QWEN2AUDIO_IMPORT_ERROR: Exception | None = None
_QWEN2AUDIO_IMPORT_ATTEMPTED: bool = False


def _get_torch(*, allow_import: bool) -> Any | None:
    global torch, _TORCH_IMPORT_ERROR, _TORCH_IMPORT_ATTEMPTED

    if torch is not None:
        return torch
    if _TORCH_IMPORT_ERROR is not None:
        return None
    if not allow_import:
        return None
    if _TORCH_IMPORT_ATTEMPTED:
        return None

    _TORCH_IMPORT_ATTEMPTED = True
    try:
        torch = importlib.import_module("torch")
        _TORCH_IMPORT_ERROR = None
    except Exception as exc:
        _TORCH_IMPORT_ERROR = exc
        torch = None
    return torch


def _get_original_whisper_model() -> Any | None:
    global _ORIGINAL_WHISPER_MODEL
    global _ORIGINAL_WHISPER_MODEL_IMPORT_ERROR
    global _ORIGINAL_WHISPER_MODEL_IMPORT_ATTEMPTED

    if _ORIGINAL_WHISPER_MODEL is not None:
        return _ORIGINAL_WHISPER_MODEL
    if _ORIGINAL_WHISPER_MODEL_IMPORT_ERROR is not None:
        return None
    if _ORIGINAL_WHISPER_MODEL_IMPORT_ATTEMPTED:
        return None

    _ORIGINAL_WHISPER_MODEL_IMPORT_ATTEMPTED = True
    try:
        module = importlib.import_module("faster_whisper")
        _ORIGINAL_WHISPER_MODEL = getattr(module, "WhisperModel")
        _ORIGINAL_WHISPER_MODEL_IMPORT_ERROR = None
    except Exception as exc:
        _ORIGINAL_WHISPER_MODEL_IMPORT_ERROR = exc
        _ORIGINAL_WHISPER_MODEL = None
    return _ORIGINAL_WHISPER_MODEL


def _get_qwen2audio_classes() -> tuple[Any, Any] | None:
    global _QWEN2AUDIO_CLASSES
    global _QWEN2AUDIO_IMPORT_ERROR
    global _QWEN2AUDIO_IMPORT_ATTEMPTED

    if _QWEN2AUDIO_CLASSES is not None:
        return _QWEN2AUDIO_CLASSES
    if _QWEN2AUDIO_IMPORT_ERROR is not None:
        return None
    if _QWEN2AUDIO_IMPORT_ATTEMPTED:
        return None

    _QWEN2AUDIO_IMPORT_ATTEMPTED = True
    try:
        module = importlib.import_module("transformers")
        _QWEN2AUDIO_CLASSES = (
            getattr(module, "AutoProcessor"),
            getattr(module, "Qwen2AudioForConditionalGeneration"),
        )
        _QWEN2AUDIO_IMPORT_ERROR = None
    except Exception as exc:
        _QWEN2AUDIO_IMPORT_ERROR = exc
        _QWEN2AUDIO_CLASSES = None
    return _QWEN2AUDIO_CLASSES


def _torch_cuda_available(*, allow_import: bool = False) -> bool:
    torch_mod = _get_torch(allow_import=allow_import)
    if torch_mod is None:
        return False
    try:
        return bool(torch_mod.cuda.is_available())
    except Exception:
        return False

#
#######################################################################################################################
# Constants
#

# Get configuration values or use defaults
media_config = loaded_config_data.get('media_processing', {}) if loaded_config_data else {}
AUDIO_TRANSCRIPTION_BUFFER_SIZE_MB = media_config.get('audio_transcription_buffer_size_mb', 10)
"""int: Maximum buffer size for audio transcription in MB."""

# Transcript cache and STT settings (env overrides config)
def _stt_cache_config():
    """
    Return the `[STT-Settings]` section from the loaded config (if any).

    This helper is shared by transcript-cache and STT-related toggles so that
    all STT knobs live under a single INI section.
    """
    try:
        return get_stt_config()
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
        logging.debug("Failed to load STT cache config, using empty dict")
        return {}


_cache_cfg = _stt_cache_config()


def _to_bool(val) -> bool:
    return is_truthy(str(val))


_env_disable = _to_bool(os.getenv("STT_DISABLE_TRANSCRIPT_CACHE", ""))
_cfg_disable = _cache_cfg.get("disable_transcript_cache", False)
PERSIST_TRANSCRIPTS_DEFAULT = not (_to_bool(_cfg_disable) or _env_disable)


def _coerce_int(val):
    try:
        return int(val)
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
        return None


def _coerce_float(val, default=None):
    try:
        return float(val)
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
        return default


def _looks_like_error_text(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    return text.strip().startswith("[Error:")

# Conservative defaults to prevent unbounded cache growth unless explicitly
# disabled via `disable_transcript_cache_pruning` or environment variables.
DEFAULT_CACHE_MAX_FILES_PER_SOURCE = 32
DEFAULT_CACHE_MAX_AGE_DAYS = 30
DEFAULT_CACHE_MAX_TOTAL_MB = 512.0

_raw_max_files = os.getenv("STT_CACHE_MAX_FILES_PER_SOURCE") or _cache_cfg.get(
    "transcript_cache_max_files_per_source"
)
_raw_max_age = os.getenv("STT_CACHE_MAX_AGE_DAYS") or _cache_cfg.get(
    "transcript_cache_max_age_days"
)
_raw_max_total = os.getenv("STT_CACHE_MAX_TOTAL_MB") or _cache_cfg.get(
    "transcript_cache_max_total_mb"
)

_max_files_val = _coerce_int(_raw_max_files)
_max_age_val = _coerce_int(_raw_max_age)
_max_total_val = _coerce_float(_raw_max_total)

# When values are absent/invalid (None), fall back to conservative defaults;
# explicit 0 or negative values are preserved so callers can disable individual
# limits if desired.
CACHE_MAX_FILES_PER_SOURCE = (
    DEFAULT_CACHE_MAX_FILES_PER_SOURCE if _max_files_val is None else _max_files_val
)
CACHE_MAX_AGE_DAYS = (
    DEFAULT_CACHE_MAX_AGE_DAYS if _max_age_val is None else _max_age_val
)
CACHE_MAX_TOTAL_MB = (
    DEFAULT_CACHE_MAX_TOTAL_MB if _max_total_val is None else _max_total_val
)

_env_skip_prevalidation = _to_bool(os.getenv("STT_SKIP_AUDIO_PREVALIDATION", ""))
_cfg_skip_prevalidation = _cache_cfg.get("skip_audio_prevalidation", False)
SKIP_AUDIO_PREVALIDATION = _to_bool(_cfg_skip_prevalidation) or _env_skip_prevalidation

_env_disable_prune = _to_bool(os.getenv("STT_DISABLE_TRANSCRIPT_CACHE_PRUNING", ""))
_cfg_disable_prune = _cache_cfg.get("disable_transcript_cache_pruning", False)
DISABLE_TRANSCRIPT_CACHE_PRUNING = _to_bool(_cfg_disable_prune) or _env_disable_prune

# Optional faster-whisper compute_type override. When unset or equal to "auto",
# Whisper models default to float16 on CUDA and int8 on CPU; when set to a
# supported faster-whisper compute_type (e.g. "int8", "int8_float16"), that
# value is passed through to the underlying WhisperModel.
WHISPER_COMPUTE_TYPE_OVERRIDE = str(
    _cache_cfg.get("whisper_compute_type", "")
).strip().lower()

def _resample_audio_if_needed(audio: np.ndarray, sample_rate: int, target_sr: int = 16000) -> np.ndarray:
    """
    Return audio at the target sample rate, resampling if necessary.

    Uses librosa when available; falls back to a simple linear interpolation so we
    can avoid a hard dependency. Always returns float32.
    """
    if sample_rate == target_sr:
        return audio.astype(np.float32, copy=False)
    try:
        import librosa
        return librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr).astype(np.float32, copy=False)
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
        logging.debug(f"Falling back to naive resample: {e}")
        ratio = float(target_sr) / float(sample_rate)
        new_len = max(1, int(round(len(audio) * ratio)))
        # Linear interpolation fallback
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32, copy=False)


def _resolve_project_root() -> Path:
    """
    Return the repository root used to anchor shared model directories.

    Preference order:
    1. A parent containing `models/` and a repo marker (`.git` or
       `pyproject.toml` with `tldw_Server_API/`).
    2. Otherwise, the outermost parent containing `models/`.
    3. Legacy depth-based fallback for unusual layouts.
    """
    current = Path(__file__).resolve()
    fallback_with_models: Path | None = None

    for parent in current.parents:
        models_dir = parent / "models"
        if not models_dir.is_dir():
            continue
        if (parent / ".git").exists() or (
            (parent / "pyproject.toml").exists() and (parent / "tldw_Server_API").is_dir()
        ):
            return parent
        fallback_with_models = parent

    if fallback_with_models is not None:
        return fallback_with_models

    parents = current.parents
    fallback_index = 5 if len(parents) > 5 else len(parents) - 1
    return parents[fallback_index]


PROJECT_ROOT_DIR = _resolve_project_root()
WHISPER_MODEL_BASE_DIR = (PROJECT_ROOT_DIR / "models" / "Whisper").resolve()
TRANSCRIPT_CACHE_DIR_NAME = "transcripts_cache"

_AUDIO_VALIDATION_CACHE: dict[str, tuple] = {}
_PRUNE_DISABLED_LOGGED: bool = False


def _default_transcript_cache_root() -> Path:
    """
    Return the centralized transcript cache root under the system temp directory.
    """
    root = Path(tempfile.gettempdir())
    try:
        return root.resolve(strict=False)
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as exc:
        logging.debug(f"Failed to resolve temp dir for transcript cache: {exc}")
        return root


def _sanitize_transcription_model_name(model_name: str) -> str:
    """
    Return a filesystem-safe transcription model identifier for cache files.
    """
    return "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in str(model_name)
    )


def _assert_no_symlink(path: Path, *, label: str) -> None:
    """
    Raise ValueError if the path or any existing parent is a symlink.
    """
    for candidate in [path, *path.parents]:
        if not candidate.exists():
            continue
        try:
            if candidate.is_symlink():
                raise ValueError(f"{label} may not traverse symlinks: {candidate}")
        except OSError as exc:
            raise ValueError(f"{label} could not be validated for symlinks: {candidate}") from exc


_HUGGINGFACE_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_ALLOWED_MEDIA_BASE_DIRS: Optional[list[Path]] = None
allowed_media_base_dirs_lock = threading.Lock()


def _looks_like_windows_drive(path_str: str) -> bool:
    return len(path_str) >= 2 and path_str[1] == ":" and path_str[0].isalpha()


def _is_hf_model_id(model_name: str) -> bool:
    return bool(_HUGGINGFACE_MODEL_ID_RE.match(model_name))


def _path_is_within(path: Path, base_dir: Path) -> bool:
    try:
        path.relative_to(base_dir)
        return True
    except ValueError:
        return False


def _get_allowed_media_base_dirs() -> list[Path]:
    global _ALLOWED_MEDIA_BASE_DIRS
    with allowed_media_base_dirs_lock:
        if _ALLOWED_MEDIA_BASE_DIRS is not None:
            return list(_ALLOWED_MEDIA_BASE_DIRS)

        roots: list[Path] = []
    try:
        roots.append(Path(tempfile.gettempdir()).resolve(strict=False))
    except (OSError, PermissionError, ValueError) as exc:
        logging.debug(f"Could not resolve temp directory for allowed base dirs: {exc}")
    try:
        roots.append(DatabasePaths.get_user_db_base_dir())
    except (OSError, PermissionError, ValueError, AttributeError) as exc:
        logging.debug(f"Could not resolve USER_DB_BASE_DIR for allowed base dirs: {exc}")

    _ALLOWED_MEDIA_BASE_DIRS = roots
    return list(roots)


def _resolve_allowed_base_dir(base_dir: Path, *, label: str) -> Path:
    base_resolved = Path(base_dir).resolve(strict=False)
    if not base_resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {base_resolved}")
    _assert_no_symlink(base_resolved, label=label)

    allowed_roots = _get_allowed_media_base_dirs()
    if allowed_roots and not any(_path_is_within(base_resolved, root) for root in allowed_roots):
        allowed_str = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(
            f"{label} must be under one of the allowed base directories: {allowed_str}"
        )
    return base_resolved


def _select_allowed_base_dir_for_path(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path when base_dir is not provided")

    path_resolved = path.resolve(strict=False)
    allowed_roots = _get_allowed_media_base_dirs()
    for root in allowed_roots:
        if _path_is_within(path_resolved, root):
            return root

    allowed_str = ", ".join(str(root) for root in allowed_roots) or "<none configured>"
    raise ValueError(
        f"{label} must resolve under one of the allowed base directories: {allowed_str}"
    )


def _resolve_safe_input_path(path: Path, *, base_dir: Optional[Path], label: str) -> Path:
    if base_dir is None:
        base_dir = _select_allowed_base_dir_for_path(path, label=label)
    base_resolved = _resolve_allowed_base_dir(base_dir, label=f"{label} base directory")
    safe_path = resolve_safe_local_path(path, base_resolved)
    if safe_path is None:
        raise ValueError(f"{label} must resolve under {base_resolved}")
    _assert_no_symlink(safe_path, label=label)
    return safe_path


def _resolve_audio_input_path_for_provider(
    audio_file_path: Union[str, Path],
    *,
    base_dir: Optional[Path],
    label: str = "Audio input path",
) -> Path:
    """
    Resolve and validate an audio input path for a provider.

    Delegates to _resolve_safe_input_path for resolution and validation.

    Args:
        audio_file_path: Union[str, Path] audio input path to resolve.
        base_dir: Optional[Path] base directory constraint for validation.
        label: str label used for validation messages.

    Returns:
        Path: Resolved, validated audio input path.
    """
    return _resolve_safe_input_path(Path(audio_file_path), base_dir=base_dir, label=label)


def _normalize_whisper_model_identifier(
    model_name: str,
    *,
    base_dir: Optional[Path] = None,
) -> str:
    raw = str(model_name or "").strip()
    if not raw:
        raise ValueError("Whisper model identifier cannot be empty")

    # If this looks like a Hugging Face Hub model id and *not* a local path,
    # return it directly. We avoid interpreting it as a filesystem path.
    if (
        _is_hf_model_id(raw)
        and not raw.startswith(("/", ".", "~"))
        and not _looks_like_windows_drive(raw)
    ):
        return raw

    path_like = (
        raw.startswith(("/", ".", "~"))
        or _looks_like_windows_drive(raw)
        or os.sep in raw
        or (os.altsep and os.altsep in raw)
    )

    if not path_like:
        if raw in {".", ".."} or ".." in raw:
            raise ValueError("Whisper model identifier may not contain path traversal components")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", raw):
            raise ValueError("Whisper model identifier contains invalid characters")
        return raw

    base_root = base_dir if base_dir is not None else WHISPER_MODEL_BASE_DIR
    safe_path = resolve_safe_local_path(Path(raw), Path(base_root))
    if safe_path is None:
        raise ValueError(
            f"Whisper model path must resolve under {Path(base_root).resolve(strict=False)}"
        )
    if not safe_path.exists():
        raise ValueError(f"Whisper model path does not exist: {safe_path}")
    try:
        _assert_no_symlink(safe_path, label="Whisper model path")
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return str(safe_path)


def validate_whisper_model_identifier(model_name: str) -> str:
    """
    Validate a Whisper model identifier and return the normalized value.

    This helper rejects path-like identifiers that escape the allowed model
    root and is safe to call on user-supplied model parameters.
    """
    return _normalize_whisper_model_identifier(
        model_name,
        base_dir=WHISPER_MODEL_BASE_DIR,
    )


def _normalize_qwen2audio_model_identifier(model_id: str) -> str:
    """Validate and normalize a Qwen2Audio model identifier.

    Accepts Hugging Face model ids and local filesystem paths that resolve
    under `WHISPER_MODEL_BASE_DIR`. Paths outside the configured model root are
    rejected to prevent path traversal or arbitrary filesystem access when the
    identifier comes from configuration/user-controlled sources.
    """
    raw = str(model_id or "").strip()
    if not raw:
        raise ValueError("Qwen2Audio model identifier cannot be empty")

    # Prefer canonical Hub identifiers when possible.
    if _is_hf_model_id(raw):
        return raw

    # For local paths, require the resolved target to remain under model root.
    safe_path = resolve_safe_local_path(Path(raw), WHISPER_MODEL_BASE_DIR)
    if safe_path is None:
        raise ValueError(
            f"Qwen2Audio model path must resolve under {WHISPER_MODEL_BASE_DIR.resolve(strict=False)}"
        )
    if not safe_path.exists():
        raise ValueError(f"Qwen2Audio model path does not exist: {safe_path}")
    if not safe_path.is_dir():
        raise ValueError(f"Qwen2Audio model path is not a directory: {safe_path}")
    _assert_no_symlink(safe_path, label="Qwen2Audio model path")
    return str(safe_path)


def validate_qwen2audio_model_identifier(model_id: str) -> str:
    """Public validator for Qwen2Audio model identifiers."""
    return _normalize_qwen2audio_model_identifier(model_id)


def _resolve_whisper_download_root(download_root: Optional[Union[str, Path]]) -> Path:
    base_root = WHISPER_MODEL_BASE_DIR
    root = Path(download_root).expanduser() if download_root else base_root

    root = (base_root / root).resolve(strict=False) if not root.is_absolute() else root.resolve(strict=False)

    safe_root = resolve_safe_local_path(root, base_root)
    if safe_root is None:
        raise ValueError(
            f"Whisper download root must resolve under {base_root.resolve(strict=False)}"
        )

    _assert_no_symlink(safe_root, label="Whisper model download root")
    if safe_root.exists() and not safe_root.is_dir():
        raise ValueError(f"Whisper download root is not a directory: {safe_root}")

    safe_root.mkdir(parents=True, exist_ok=True)
    return safe_root


def _check_standard_model_under_download_root(
    identifier: str,
    download_root: Path,
) -> Optional[Path]:
    """
    Check if a standard model identifier exists under the download root.

    Returns the resolved path if found, otherwise None.
    Skips check if identifier looks like a filesystem path.
    """
    if (
        identifier.startswith(("/", ".", "~"))
        or _looks_like_windows_drive(identifier)
        or os.sep in identifier
        or (os.altsep and os.altsep in identifier)
    ):
        logging.info(
            "Standard model identifier %r looks like a path; skipping local "
            "directory check under download_root.",
            identifier,
        )
        return None

    candidate = download_root / identifier
    try:
        root_resolved = download_root.resolve()
        candidate_resolved = candidate.resolve()
        candidate_resolved.relative_to(root_resolved)
    except (ValueError, OSError):
        return None

    if candidate_resolved.is_dir():
        logging.info(
            "Found standard model '%s' in custom download root: %s",
            identifier,
            candidate_resolved,
        )
        return candidate_resolved

    logging.info(
        "Standard model '%s' not in custom root. Passing name to faster-whisper.",
        identifier,
    )
    return None


def _resolve_transcript_cache_dir(
    audio_path: Path,
    *,
    base_dir: Optional[Path] = None,
) -> Path:
    """
    Resolve and create the transcript cache directory.

    When base_dir is provided, creates the cache under that directory.
    Otherwise, uses the system temp root for centralized caching.
    """
    root_dir = base_dir if base_dir is not None else _default_transcript_cache_root()
    cache_dir = root_dir / TRANSCRIPT_CACHE_DIR_NAME
    _assert_no_symlink(cache_dir, label="Transcript cache directory")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def prune_transcript_cache(
    cache_dir: Path,
    *,
    current_file: Optional[Path] = None,
    max_age_days: Optional[int] = CACHE_MAX_AGE_DAYS,
    max_total_mb: Optional[float] = CACHE_MAX_TOTAL_MB,
    max_files_per_source: Optional[int] = CACHE_MAX_FILES_PER_SOURCE,
) -> None:
    """
    Prune cached transcript files by age, total size, and per-source limits.

    Args:
        cache_dir: Directory containing transcript cache files.
        current_file: File that was just written; always preserved.
        max_age_days: Delete files older than this many days (None disables).
        max_total_mb: Delete oldest files until total size is under limit (None disables).
        max_files_per_source: Keep only the newest N files per base source (None disables).
    """
    if DISABLE_TRANSCRIPT_CACHE_PRUNING:
        global _PRUNE_DISABLED_LOGGED
        if not _PRUNE_DISABLED_LOGGED:
            _PRUNE_DISABLED_LOGGED = True
            logging.info(
                "Transcript cache pruning is disabled via STT-Settings.disable_transcript_cache_pruning "
                "or STT_DISABLE_TRANSCRIPT_CACHE_PRUNING; cached transcripts may grow without bound."
            )
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
        return

    files = sorted(
        cache_dir.glob("*-whisper_model-*.segments*.json"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    if current_file:
        files = [f for f in files if f.exists()]

    now = datetime.now().timestamp()

    def _unlink(path: Path):
        try:
            path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
            pass

    # Age-based pruning
    if max_age_days is not None and max_age_days >= 0:
        cutoff = now - (max_age_days * 86400)
        for f in list(files):
            if f == current_file:
                continue
            try:
                if f.stat().st_mtime < cutoff:
                    _unlink(f)
                    files.remove(f)
            except FileNotFoundError:
                files.remove(f)

    # Per-source limit (group by base name before ".segments")
    if max_files_per_source is not None and max_files_per_source > 0:
        grouped: dict[str, list[Path]] = {}
        for f in files:
            key = f.name.split(".segments")[0]
            grouped.setdefault(key, []).append(f)
        for paths in grouped.values():
            # paths already sorted newest→oldest
            for old in paths[max_files_per_source:]:
                if old == current_file:
                    continue
                _unlink(old)
                files.remove(old)

    # Total size pruning
    if max_total_mb is not None and max_total_mb > 0:
        def _file_size(path: Path) -> int:
            try:
                return path.stat().st_size
            except FileNotFoundError:
                return 0

        # Re-sort oldest→newest for eviction
        files_sorted_oldest = sorted(files, key=lambda p: p.stat().st_mtime if p.exists() else 0)
        total_bytes = sum(_file_size(f) for f in files)
        limit_bytes = max_total_mb * 1024 * 1024
        idx = 0
        while total_bytes > limit_bytes and idx < len(files_sorted_oldest):
            victim = files_sorted_oldest[idx]
            idx += 1
            if victim == current_file:
                continue
            total_bytes -= _file_size(victim)
            _unlink(victim)

#######################################################################################################################
# Function Definitions
#

# Convert video .m4a into .wav using ffmpeg
#   ffmpeg -i "example.mp4" -ar 16000 -ac 1 -c:a pcm_s16le "output.wav"
#       https://www.gyan.dev/ffmpeg/builds/
#

# FIXME
# 1. Implement chunking for large audio files
# def speech_to_text(audio_file_path, selected_source_lang='en', whisper_model='medium.en', vad_filter=False, chunk_size=30):
#     # ... existing code ...
#     segments = []
#     for segment_chunk in whisper_model_instance.transcribe(audio_file_path, beam_size=10, best_of=10, vad_filter=vad_filter, chunk_size=chunk_size):
#         # Process each chunk
#         # ... existing code ...
#
# 2. Use generators
#     def generate_segments(segments_raw):
#         for segment_chunk in segments_raw:
#             yield {
#                 "Time_Start": segment_chunk.start,
#                 "Time_End": segment_chunk.end,
#                 "Text": segment_chunk.text
#             }
#     # Usage
#     segments = list(generate_segments(segments_raw))
#
# 3. Use subprocess instead of os.system for ffmpeg
# 4. Adjust CPU threads properly
# 5. Use quantized models - compute_type="int8"


def perform_transcription(
    video_path: str,
    offset: int,
    transcription_model: str,
    vad_use: bool,
    diarize: bool = False,
    overwrite: bool = False,
    transcription_language: str = 'en',
    hotwords: Optional[Sequence[str] | str] = None,
    temp_dir: Optional[str] = None,
    end_seconds: Optional[int] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    ):
    """
    Converts a video or audio file to WAV format, performs transcription,
    and optionally attempts diarization (currently a placeholder).

    The function handles file existence checks and error management throughout
    the process. If diarization is requested, it's currently a non-functional
    placeholder and will likely return with the audio path but no segments.

    Args:
        video_path: The file path to the video or audio file to be processed.
        offset: The time offset (in seconds) from the beginning of the
            media file from which to start processing. This is passed to
            `convert_to_wav`.
        transcription_model: The name or path of the transcription model
            to be used (e.g., 'base.en', 'large-v3').
        vad_use: A boolean indicating whether to use Voice Activity Detection
            (VAD) during transcription.
        diarize: A boolean indicating whether to perform speaker diarization.
            (Note: Diarization functionality is currently a FIXME placeholder
            and not fully implemented).
        overwrite: A boolean indicating whether to overwrite existing
            transcription files. If False and a relevant transcription file
            exists, it will be loaded.
        transcription_language: The language code (e.g., 'en', 'es') for
            transcription. Defaults to 'en'.
        hotwords: Optional hotword hints. Accepts a list/sequence or a JSON/CSV
            string. This is primarily used by VibeVoice-ASR and ignored by other
            providers.
        temp_dir: An optional path to a temporary directory. When provided,
            the input and output paths must resolve under this directory.
        end_seconds: Optional absolute end time (in seconds) for transcription.
            When provided, audio is clipped to the interval [offset, end_seconds).
        cancel_check: Optional callable that returns True when processing should
            be cancelled.

    Returns:
        A tuple containing:
        - `Optional[str]`: The file path to the converted WAV audio file.
          This can be `None` if the initial conversion fails.
        - `Optional[list]`: A list of transcription segments. Each segment
          is typically a dictionary with 'start_seconds', 'end_seconds',
          and 'Text'. This can be `None` if transcription (or diarization,
          if attempted) fails or if an existing invalid file is encountered
          with `overwrite=False`.

        Specific return scenarios:
        - `(audio_file_path, segments_list)`: On successful conversion and transcription.
        - `(None, None)`: On critical failure (e.g., media conversion fails).
        - `(audio_file_path, None)`: If conversion succeeds but transcription fails,
          or if diarization is attempted and fails (as it's a placeholder).
    """
    audio_file_path = None  # Track generated WAV path even if conversion fails early
    try:
        logging.info(f"Initiating transcription process for: {video_path}")
        base_dir_path = None
        if temp_dir:
            try:
                base_dir_path = _resolve_allowed_base_dir(
                    Path(temp_dir),
                    label="Audio temp directory",
                )
            except ValueError as exc:
                logging.error(f"Invalid temp_dir for transcription: {exc}")
                return None, None
        _check_cancel(cancel_check, label="transcription preflight")
        # 1. Convert to WAV - Catch ConversionError specifically
        try:
            if base_dir_path is not None:
                safe_video_path = resolve_safe_local_path(
                    Path(video_path),
                    base_dir_path,
                )
                if safe_video_path is None:
                    logging.error(
                        f"Audio input path rejected outside temp_dir: {video_path}"
                    )
                    return None, None
                video_path = str(safe_video_path)
            audio_file_path = convert_to_wav(
                video_path,
                offset=offset,
                end_time=end_seconds,
                overwrite=overwrite,
                base_dir=base_dir_path,
                cancel_check=cancel_check,
            )
            if not audio_file_path or not os.path.exists(audio_file_path):
                 # This case might occur if convert_to_wav returns None/empty path without raising error
                 logging.error(f"Conversion to WAV failed or produced no file for {video_path}")
                 return None, None # Critical failure
            logging.debug(f"Converted audio file path: {audio_file_path}")
        except ConversionError as e:
            logging.error(f"Audio conversion failed for {video_path}: {e}")
            return None, None # Critical failure, stop processing

        # 2. Define paths
        audio_path_obj = Path(audio_file_path)
        if base_dir_path is not None:
            safe_audio_path = resolve_safe_local_path(
                audio_path_obj,
                base_dir_path,
            )
            if safe_audio_path is None:
                logging.error(
                    f"Audio output path rejected outside temp_dir: {audio_file_path}"
                )
                return None, None
            audio_path_obj = safe_audio_path
            audio_file_path = str(safe_audio_path)
        _assert_no_symlink(audio_path_obj, label="Audio output path")
        cache_dir = _resolve_transcript_cache_dir(audio_path_obj, base_dir=base_dir_path)
        transcription_model_sanitized = _sanitize_transcription_model_name(transcription_model)
        hotwords_suffix = _hotwords_cache_suffix(hotwords)
        segments_json_path = (
            cache_dir
            / (
                f"{audio_path_obj.stem}-transcription_model-"
                f"{transcription_model_sanitized}{hotwords_suffix}.segments.json"
            )
        )
        diarized_json_path = (
            cache_dir
            / (
                f"{audio_path_obj.stem}-transcription_model-"
                f"{transcription_model_sanitized}{hotwords_suffix}.diarized.json"
            )
        )

        # --- Perform Diarization and Combination (if requested) ---
        if diarize:
            logging.info(f"Processing with diarization for {audio_file_path}")

            diarization_config = load_diarization_config()
            backend = str(diarization_config.get("backend", "embedding") or "embedding").lower()

            # Only require embedding diarization deps when using embedding backend.
            if not DIARIZATION_AVAILABLE and backend != "nemo_multitalk":
                logging.warning(
                    "Diarization requested but not available. Install with: pip install tldw-server[diarization]"
                )
                logging.info("Falling back to regular transcription without diarization")
                diarize = False  # Fall back to regular transcription

            else:
                # Check if diarized file already exists
                if os.path.exists(diarized_json_path) and not overwrite:
                    _assert_no_symlink(diarized_json_path, label="Diarized transcript cache file")
                    logging.info(f"Diarized file already exists (overwrite=False): {diarized_json_path}")
                    try:
                        with open(diarized_json_path, encoding='utf-8') as file:
                            loaded_data = json.load(file)
                        if isinstance(loaded_data, dict) and "segments" in loaded_data:
                            segments = loaded_data["segments"]
                        elif isinstance(loaded_data, list):
                            segments = loaded_data
                        else:
                            raise ValueError("Diarized JSON structure is not valid")

                        # Basic validation
                        if isinstance(segments, list) and all(isinstance(s, dict) and 'Text' in s for s in segments):
                            logging.debug("Loaded valid diarized segments from existing file.")
                            return audio_file_path, segments
                        else:
                            logging.warning(f"Existing diarized file {diarized_json_path} has invalid format")
                    except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
                        logging.warning(f"Failed to read/parse existing diarized file: {e}")
                        # Continue to regenerate

                # Optional: NeMo multitalk diarization (coupled Parakeet ASR)
                if backend == "nemo_multitalk":
                    provider = None
                    variant = None
                    try:
                        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (
                            get_stt_provider_registry,
                        )

                        registry = get_stt_provider_registry()
                        provider, _, variant = registry.resolve_provider_for_model(transcription_model or "")
                    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as exc:
                        logging.warning(f"Unable to resolve STT provider for multitalk diarization: {exc}")

                    if provider != "parakeet":
                        logging.error(
                            "NeMo multitalk diarization requires Parakeet STT provider; disabling diarization."
                        )
                        diarize = False
                    elif variant not in (None, "standard"):
                        raw_model = str(transcription_model or "").strip().lower()
                        # If the user did not explicitly request a variant, allow multitalk by
                        # forcing the NeMo-standard path (tests expect base "parakeet" to work).
                        if raw_model == "parakeet":
                            logging.info(
                                "NeMo multitalk diarization requires Parakeet 'standard' variant; "
                                "overriding base 'parakeet' to standard for multitalk."
                            )
                            variant = "standard"
                        else:
                            logging.error(
                                "NeMo multitalk diarization only supports Parakeet 'standard' (NeMo) variant; "
                                f"got '{variant}'. Disabling diarization."
                            )
                            diarize = False
                    if diarize and provider == "parakeet":
                        try:
                            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Diarization_Nemo_Multitalk import (
                                transcribe_with_nemo_multitalk,
                            )

                            diarized_result = transcribe_with_nemo_multitalk(
                                audio_file_path,
                                diarization_config,
                                output_path=str(diarized_json_path),
                            )

                            if isinstance(diarized_result, dict) and diarized_result.get("segments"):
                                final_segments = diarized_result["segments"]
                                _assert_no_symlink(diarized_json_path, label="Diarized transcript cache file")
                                with open(diarized_json_path, 'w', encoding='utf-8') as f:
                                    json.dump({'segments': final_segments}, f, ensure_ascii=False, indent=2)

                                logging.info(f"NeMo multitalk diarization successful. Saved to {diarized_json_path}")
                                return audio_file_path, final_segments

                            logging.warning("NeMo multitalk diarization returned no segments; falling back.")
                        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as exc:
                            logging.error(f"NeMo multitalk diarization failed: {exc}")

                if diarize:
                    # First, get the transcription segments via the unified STT helper
                    logging.info("Generating transcription for diarization")
                    artifact = run_stt_batch_via_registry(
                        audio_file_path,
                        transcription_model,
                        vad_filter=vad_use,
                        selected_source_lang=transcription_language,
                        hotwords=hotwords,
                        base_dir=base_dir_path,
                        cancel_check=cancel_check,
                    )
                    transcription_segments = artifact.get("segments") or []

                    if transcription_segments is None:
                        logging.error(f"Transcription generation failed for {audio_file_path}")
                        return audio_file_path, None

                    # Now perform diarization
                    try:
                        logging.info("Performing speaker diarization...")
                        diarization_service = DiarizationService()

                        if not diarization_service.is_available:
                            logging.warning("Diarization service is not available (missing dependencies)")
                            logging.info("Returning transcription without speaker labels")
                            return audio_file_path, transcription_segments

                        # Perform diarization with transcription segments
                        diarized_segments = diarization_service.diarize(
                            audio_path=audio_file_path,
                            transcription_segments=transcription_segments
                        )

                        if diarized_segments and 'segments' in diarized_segments:
                            final_segments = diarized_segments['segments']

                            # Save diarized results
                            _assert_no_symlink(diarized_json_path, label="Diarized transcript cache file")
                            with open(diarized_json_path, 'w', encoding='utf-8') as f:
                                json.dump({'segments': final_segments}, f, ensure_ascii=False, indent=2)

                            logging.info(f"Diarization successful. Saved to {diarized_json_path}")
                            return audio_file_path, final_segments
                        else:
                            logging.warning("Diarization returned no segments")
                            return audio_file_path, transcription_segments

                    except DiarizationError as e:
                        logging.error(f"Diarization failed for {audio_file_path}: {e}")
                        logging.warning("Proceeding with transcription only due to diarization error.")
                        # Add a warning to the first segment if possible
                        if transcription_segments:
                            transcription_segments[0]['Text'] = "[Note: Speaker diarization failed] " + transcription_segments[0].get('Text', '')
                        return audio_file_path, transcription_segments

                    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
                        logging.error(f"Unexpected error during diarization: {e}", exc_info=True)
                        return audio_file_path, transcription_segments

        # If we get here and diarize was set to False (either originally or as fallback), continue with regular path

        # 4. Handle Non-Diarized Path
        if not diarize:
            logging.info(f"Processing without diarization for {audio_file_path}")
            # Check if non-diarized JSON exists
            if os.path.exists(segments_json_path) and not overwrite:
                _assert_no_symlink(segments_json_path, label="Transcript cache file")
                logging.info(f"Segments file already exists (overwrite=False): {segments_json_path}")
                try:
                    with open(segments_json_path, encoding='utf-8') as file:
                        loaded_data = json.load(file)
                    # Handle potential structures: {'segments': [...]} or just [...]
                    if isinstance(loaded_data, dict) and "segments" in loaded_data:
                        segments = loaded_data["segments"]
                    elif isinstance(loaded_data, list):
                        segments = loaded_data
                    else:
                        raise ValueError("JSON structure is not a list or {'segments': list}")

                    # Basic validation
                    if isinstance(segments, list) and all(isinstance(s, dict) and 'Text' in s for s in segments):
                        logging.debug("Loaded valid segments from existing file.")
                        return audio_file_path, segments
                    else:
                        logging.warning(f"Existing segments file {segments_json_path} has invalid format, regenerating.")
                except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
                    logging.warning(f"Failed to read/parse existing segments file {segments_json_path}: {e}. Regenerating.")

            # Generate new transcription (or overwrite existing)
            logging.info(f"Generating/Overwriting transcription for {audio_file_path}")
            artifact = run_stt_batch_via_registry(
                audio_file_path,
                transcription_model,
                vad_filter=vad_use,
                selected_source_lang=transcription_language,
                hotwords=hotwords,
                base_dir=base_dir_path,
                cancel_check=cancel_check,
            )
            segments = artifact.get("segments") or []
            if not segments:
                logging.error(f"Transcription generation failed for {audio_file_path} (no segments)")
                return audio_file_path, None  # Return path, None segments

            # Persist a cache artifact keyed by model + hotwords to avoid
            # cross-contaminating hotword-guided transcripts.
            try:
                _assert_no_symlink(segments_json_path, label="Transcript cache file")
                with open(segments_json_path, "w", encoding="utf-8") as f:
                    json.dump({"segments": segments}, f, ensure_ascii=False, indent=2)
            except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as cache_err:
                logging.debug(f"Failed to persist transcript cache artifact: {cache_err}")

            logging.info(f"Successfully generated/loaded transcription for {audio_file_path}")
            return audio_file_path, segments

    except TranscriptionCancelled:
        raise
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
        # Catch-all for unexpected errors during the process
        logging.error(f"Unexpected error in perform_transcription for {video_path}: {e}", exc_info=True)
        # If conversion succeeded, return path, else None. Always return None for segments on error.
        return (audio_file_path, None) if audio_file_path else (None, None)


def re_generate_transcription(
    audio_file_path,
    whisper_model,
    vad_filter,
    selected_source_lang='en',
    base_dir: Optional[Path] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
):
    """
    Calls `speech_to_text` to perform transcription on an audio file and handles potential errors.

    This function serves as a wrapper around `speech_to_text`, primarily for
    regenerating transcriptions. It ensures that all necessary parameters are
    passed to `speech_to_text` and catches exceptions that might occur during
    the transcription process.

    Args:
        audio_file_path: The path to the audio file to be transcribed.
        whisper_model: The name or path of the Whisper model to use for transcription.
        vad_filter: A boolean indicating whether to use Voice Activity Detection (VAD).
        selected_source_lang: The language code for the source audio (e.g., 'en', 'es').
            Defaults to 'en'.
        base_dir: Optional base directory used to validate the audio file path.
        cancel_check: Optional callable that returns True when processing should be cancelled.

    Returns:
        A tuple containing:
        - `str`: The `audio_file_path` that was processed.
        - `Optional[list]`: A list of transcription segments if successful,
          or `None` if transcription fails or yields no segments. Each segment is
          a dictionary, typically with 'start_seconds', 'end_seconds', and 'Text'.
    """
    logging.info(f"Regenerating transcription for {audio_file_path} using model {whisper_model}")
    try:
        # IMPORTANT: Pass all necessary parameters to speech_to_text.
        # The canonical return type of speech_to_text is a list of segment
        # dicts (or (segments, language) when return_language=True). We treat
        # any dict-with-'segments' shape here as a defensive normalization
        # only, not as a public contract.
        segments = speech_to_text(
            audio_file_path,
            whisper_model=whisper_model,
            selected_source_lang=selected_source_lang,  # Ensure language is passed
            vad_filter=vad_filter,
            diarize=False,  # Explicitly false for non-diarized regeneration
            base_dir=base_dir,
            cancel_check=cancel_check,
        )
        # speech_to_text returns the segments list directly on success (or raises).
        # Normalize a dict-with-'segments' defensively in case a future change or
        # external wrapper passes that shape through.
        if isinstance(segments, dict) and 'segments' in segments:
            actual_segments = segments['segments']
        else:
            actual_segments = segments  # Assuming it returns the list directly now or handles errors by raising

        if not actual_segments:
            logging.warning(f"Re-generation yielded no segments for {audio_file_path}")
            return audio_file_path, None  # Return path but None segments on empty result

        logging.info(f"Successfully re-generated transcription for {audio_file_path}")
        return audio_file_path, actual_segments
    except TranscriptionCancelled:
        raise
    except RuntimeError as e:
        logging.error(f"RuntimeError during re_generate_transcription for {audio_file_path}: {e}")
        return audio_file_path, None  # Return path but None segments on error
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Unexpected error during re_generate_transcription for {audio_file_path}: {e}", exc_info=True)
        return audio_file_path, None  # Return path but None segments on error


#####################################
# Memory-Saving Indefinite Recording
#####################################

class PartialTranscriptionThread(threading.Thread):
    """
    A thread that performs partial (live) transcriptions on audio chunks.

    This thread consumes audio data from a queue, maintains a rolling buffer
    of recent audio, and periodically attempts to transcribe this buffer to
    provide near real-time transcription updates.

    Attributes:
        audio_queue (queue.Queue): Queue to get audio chunks from.
        stop_event (threading.Event): Event to signal the thread to stop.
        partial_text_state (dict): A dictionary (shared state, needs locking)
            to store the latest partial transcription text under the key "text".
        lock (threading.Lock): Lock to protect access to `partial_text_state`.
        live_model (str): The transcription model to use for partial transcriptions.
        sample_rate (int): The sample rate of the input audio (Hz).
        channels (int): The number of audio channels.
        partial_update_interval (float): How often (in seconds) to attempt a
            partial transcription.
        partial_chunk_seconds (float): The duration (in seconds) of the audio
            rolling buffer to keep in memory for partial transcription.
        exception_encountered (Optional[Exception]): Stores any exception that
            occurs during the `run` method.
    """
    def __init__(
        self,
        audio_queue: queue.Queue,
        stop_event: threading.Event,
        partial_text_state: dict,
        lock: threading.Lock,
        live_model: str,          # model for partial
        sample_rate=44100,
        channels=2,
        partial_update_interval=2.0,   # how often we attempt a partial transcription
        partial_chunk_seconds=5,
    ):
        """
        Initializes the PartialTranscriptionThread.

        Args:
            audio_queue: Queue for incoming audio data chunks (bytes).
            stop_event: Event to signal termination of the thread.
            partial_text_state: Dictionary to store the output partial transcription.
                Must be thread-safe if accessed from outside without the lock.
            lock: A threading.Lock instance to synchronize access to `partial_text_state`.
            live_model: Name or path of the transcription model for live updates.
            sample_rate: Expected sample rate of the audio in Hz.
            channels: Number of audio channels.
            partial_update_interval: Interval in seconds between transcription attempts.
            partial_chunk_seconds: Maximum duration of audio (in seconds) to hold in
                the rolling buffer for partial transcription.
        """
        super().__init__(daemon=True)
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.partial_text_state = partial_text_state
        self.lock = lock
        self.live_model = live_model

        self.sample_rate = sample_rate
        self.channels = channels
        self.partial_update_interval = partial_update_interval
        self.partial_chunk_seconds = partial_chunk_seconds

        # Rolling buffer for partial
        self.audio_buffer = b""
        # We only keep last X seconds in memory for partial
        self.max_partial_bytes = int(self.partial_chunk_seconds * self.sample_rate * self.channels * 2)
        # Also enforce a hard limit based on configuration
        max_buffer_bytes = AUDIO_TRANSCRIPTION_BUFFER_SIZE_MB * 1024 * 1024
        self.max_partial_bytes = min(self.max_partial_bytes, max_buffer_bytes)

        self.last_ts = time.time()

        # Keep track of any exceptions
        self.exception_encountered = None

    def run(self):
        """
        Main loop for the partial transcription thread.

        Continuously reads audio data from the queue, appends it to a
        rolling buffer, and periodically transcribes the buffer content.
        Updates `self.partial_text_state` with the latest transcription.
        If an error occurs during transcription, it's stored in
        `self.exception_encountered`.
        """
        while not self.stop_event.is_set():
            now = time.time()
            if now - self.last_ts < self.partial_update_interval:
                time.sleep(0.1)
                continue

            # Gather new chunks from the queue
            new_data = []
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get_nowait()
                new_data.append(chunk)

            if new_data:
                combined_new_data = b"".join(new_data)
                # Append to rolling buffer
                self.audio_buffer += combined_new_data

                # Enforce maximum partial buffer size
                if len(self.audio_buffer) > self.max_partial_bytes:
                    self.audio_buffer = self.audio_buffer[-self.max_partial_bytes:]

            # If rolling buffer is large enough, do partial transcription
            if len(self.audio_buffer) > (self.sample_rate * self.channels * 2):  # ~1s
                try:
                    # Convert from 16-bit PCM to float32
                    audio_np = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

                    # If channels=2, you may want to downmix to mono:
                    # If your STT supports stereo, skip this step.
                    if self.channels == 2:
                        audio_np = audio_np.reshape((-1, 2))
                        audio_np = np.mean(audio_np, axis=1)  # simple stereo -> mono

                    # Transcribe using configured provider or default via the
                    # shared STT provider registry. This keeps default selection
                    # consistent with the rest of the STT module and the PRD.
                    try:
                        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (  # type: ignore
                            get_stt_provider_registry,
                        )

                        provider = get_stt_provider_registry().get_default_provider_name()
                    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
                        # Defensive fallback in case the registry cannot be
                        # imported in a constrained environment.
                        config = loaded_config_data or load_and_log_configs()
                        provider = "faster-whisper"
                        if config and 'STT-Settings' in config:
                            provider = config['STT-Settings'].get('default_transcriber', 'faster-whisper')

                    if provider == 'parakeet':
                        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
                            transcribe_with_parakeet,
                        )
                        variant = config['STT-Settings'].get('nemo_model_variant', 'standard') if config else 'standard'
                        partial_text = transcribe_with_parakeet(audio_np, self.sample_rate, variant)
                    elif provider == 'canary':
                        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
                            transcribe_with_canary,
                        )
                        partial_text = transcribe_with_canary(audio_np, self.sample_rate, "en")
                    else:
                        partial_text = transcribe_audio(
                            audio_np,
                            sample_rate=self.sample_rate,
                            whisper_model=self.live_model,
                            speaker_lang="en",
                            transcription_provider=provider
                        )

                    with self.lock:
                        self.partial_text_state["text"] = partial_text
                except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
                    self.exception_encountered = e
                    logging.error(f"Partial transcription error: {e}")

            self.last_ts = time.time()


##########################################################
# Transcription Sink Function
def transcribe_audio(audio_data: np.ndarray, transcription_provider, sample_rate: int = 16000, speaker_lang=None, whisper_model="distil-large-v3") -> str:
    """
    Canonical waveform-based entry point for speech-to-text across providers.

    This helper is the central sink used by higher-level modules (REST endpoints,
    speech chat, live tools) when they already have in-memory audio. It selects
    the transcription provider based on the `transcription_provider` argument or
    a default from configuration and normalizes results to a plain text string.

    It currently supports 'qwen2audio', 'parakeet', 'canary', 'external:*', and
    'faster-whisper'. For 'faster-whisper', it routes to `speech_to_text` and
    merges the returned segments into a single user-facing transcript.

    Args:
        audio_data: A NumPy array containing the raw audio waveform (float32).
        transcription_provider: The name of the transcription provider to use
            (e.g., 'qwen2audio', 'parakeet', 'faster-whisper'). If None,
            the default is loaded from configuration.
        sample_rate: The sample rate of the `audio_data` in Hz.
        speaker_lang: The language code of the audio (e.g., 'en', 'es').
            Used by faster-whisper. If None, language detection may be attempted
            by the underlying model.
        whisper_model: The specific model name or path to use if 'faster-whisper'
            is the provider (e.g., 'distil-large-v3', 'base.en').

    Returns:
        The transcribed text as a string. This function never returns segments;
        all providers are normalized to a single text transcript.

        For providers that may fail (e.g. Qwen2Audio, external backends), this
        function returns a provider-specific error sentinel such as
        "[Transcription error] ...". Callers that surface `transcribe_audio`
        output to users must treat such sentinels as structured errors via
        `is_transcription_error_message` rather than as real user speech.

    Notes:
        - `transcribe_audio` is the preferred entry point whenever you already
          have a NumPy waveform (speech chat, WebSocket sinks, background audio
          tools).
        - `speech_to_text` is the canonical file/segment-based entry point used
          by media ingestion and offline workers; it returns structured segments
          (or `(segments, language)` when requested) rather than plain text.
    """
    # Load STT settings safely; fall back to sane defaults if missing/malformed.
    try:
        stt_cfg = get_stt_config()
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
        stt_cfg = {}

    if not transcription_provider:
        # Load default transcription provider via config file, but guard against
        # missing sections/keys so we always have a sane default.
        transcription_provider = stt_cfg.get("default_transcriber", "faster-whisper")

    if transcription_provider.lower() == 'qwen2audio':
        logging.info("Transcribing using Qwen2Audio")
        try:
            return transcribe_with_qwen2audio(audio_data, sample_rate)
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
            logging.error(f"Qwen2Audio transcription failed: {e}", exc_info=True)
            return f"[Transcription error] Qwen2Audio transcription failed: {e}"

    elif transcription_provider.lower() == "parakeet":
        logging.info("Transcribing using Parakeet")
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
                transcribe_with_parakeet,
            )
            # Get model variant from config
            variant = stt_cfg.get('nemo_model_variant', 'standard')

            return transcribe_with_parakeet(audio_data, sample_rate, variant)
        except ImportError as e:
            logging.error(f"Failed to import Nemo transcription module: {e}")
            return "Nemo transcription module not available. Please check installation."
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
            logging.error(f"Parakeet transcription failed: {e}")
            return f"Parakeet transcription error: {str(e)}"

    elif transcription_provider.lower() == "canary":
        logging.info("Transcribing using Canary")
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
                transcribe_with_canary,
            )
            return transcribe_with_canary(audio_data, sample_rate, speaker_lang)
        except ImportError as e:
            logging.error(f"Failed to import Nemo transcription module: {e}")
            return "Nemo transcription module not available. Please check installation."
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
            logging.error(f"Canary transcription failed: {e}")
            return f"Canary transcription error: {str(e)}"

    elif transcription_provider.lower() == "external" or transcription_provider.lower().startswith("external:"):
        logging.info("Transcribing using external provider")
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (
                transcribe_with_external_provider,
            )
            # Check if a specific provider is specified (e.g., "external:myapi")
            provider_name = "default"
            if ":" in transcription_provider:
                provider_name = transcription_provider.split(":", 1)[1]

            return transcribe_with_external_provider(
                audio_data,
                sample_rate=sample_rate,
                provider_name=provider_name,
                language=speaker_lang
            )
        except ImportError as e:
            logging.error(f"Failed to import external provider module: {e}")
            return "External provider module not available. Please check installation."
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
            logging.error(f"External provider transcription failed: {e}")
            return f"External provider transcription error: {str(e)}"

    else:
        logging.info(f"Transcribing using faster-whisper with model: {whisper_model}")

        segments = speech_to_text(
            audio_data,
            whisper_model=whisper_model,
            selected_source_lang=speaker_lang,
            input_sample_rate=sample_rate,
        )
        if isinstance(segments, dict) and 'error' in segments:
            # handle error
            return f"Error in transcription: {segments['error']}"

        # Merge all segment texts
        final_text = " ".join(seg["Text"] for seg in segments['segments']) if isinstance(segments, dict) else " ".join(
            seg["Text"] for seg in segments)
        return final_text

#
# End of Sink Function
##########################################################


def is_transcription_error_message(msg: str) -> bool:
    """
    Heuristic to detect transcription error sentinel strings.

    This centralizes detection used by API endpoints and speech chat so that
    provider-specific error messages stay in sync with callers.
    """
    if not isinstance(msg, str):
        return False

    lower_msg = msg.lower().strip()
    if not lower_msg:
        return False

    return (
        lower_msg.startswith("[error")
        or lower_msg.startswith("[transcription error")
        or lower_msg.startswith("error in transcription")
        or lower_msg.startswith("canary transcription error")
        or lower_msg.startswith("parakeet transcription error")
        or lower_msg.startswith("external provider transcription error")
        or lower_msg.startswith("external provider module not available")
        or lower_msg.startswith("external provider transcription failed")
        or lower_msg.startswith("nemo transcription module not available")
        or lower_msg.startswith("failed to import nemo")
        or lower_msg.startswith("failed to import external provider")
    )


def strip_whisper_metadata_header(segments):
    """
    Remove the Whisper metadata header from the first segment's Text, if present.

    The speech_to_text function prepends a header like:
        "This text was transcribed using whisper model: ...\\n"
        "Detected language: ...\\n\\n"
    into the first segment. For API/user-facing flows we often want the bare
    transcript, so this helper trims that header in-place.

    Args:
        segments: Either a list of segment dicts or a dict with a "segments" list.

    Returns:
        The same segments object that was passed in (mutated if a header was removed).
    """
    try:
        seg_list = segments.get("segments") or [] if isinstance(segments, dict) else segments

        if not seg_list:
            return segments

        first = seg_list[0]
        if not isinstance(first, dict):
            return segments

        text = first.get("Text")
        if not isinstance(text, str):
            return segments

        header_prefix = "This text was transcribed using whisper model:"
        if not text.startswith(header_prefix):
            return segments

        # Preferred: split on the blank line separating header from content
        parts = text.split("\n\n", 1)
        if len(parts) == 2:
            first["Text"] = parts[1]
            return segments

        # Fallback: drop the first two lines (model + language)
        lines = text.splitlines()
        if len(lines) >= 3:
            first["Text"] = "\n".join(lines[2:])

        return segments
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
        # Never fail the caller because of header stripping
        return segments


def to_normalized_stt_artifact(
    text: str,
    segments,
    *,
    language: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    diarization_enabled: bool = False,
    diarization_speakers: Optional[int] = None,
) -> dict[str, Any]:
    """
    Build a normalized STT artifact from text/segments.

    This helper is used by ingestion/Jobs to ensure a consistent internal
    representation of STT results, even when different code paths
    (REST, media ingestion, workers) assemble transcripts differently.
    """
    # Normalize segments into a list
    seg_list: list[dict[str, Any]] = []
    try:
        if isinstance(segments, dict):
            maybe = segments.get("segments")
            if isinstance(maybe, list):
                seg_list = maybe
        elif isinstance(segments, list):
            seg_list = segments
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
        seg_list = []

    duration_ms: Optional[int] = None
    if duration_seconds is not None:
        try:
            duration_ms = round(max(float(duration_seconds), 0.0) * 1000)
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
            duration_ms = None

    return {
        "text": text or "",
        "language": language,
        "segments": seg_list,
        "diarization": {"enabled": bool(diarization_enabled), "speakers": diarization_speakers},
        "usage": {"duration_ms": duration_ms, "tokens": None},
        "metadata": {
            "provider": provider or "",
            "model": model or "",
        },
    }


@lru_cache(maxsize=1)
def _valid_whisper_model_sizes_for_jobs() -> set:
    """
    Cached lookup of known faster-whisper model sizes for jobs helper.

    Mirrors the audio endpoint's _valid_whisper_model_sizes but is defined
    here to avoid circular imports between core audio libs and API modules.
    """
    try:
        return set(getattr(WhisperModel, "valid_model_sizes", []))
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
        # If WhisperModel is unavailable, fall back to empty set so that
        # model mapping simply treats all inputs as aliases.
        return set()


def _map_openai_audio_model_to_whisper_for_jobs(model: Optional[str]) -> str:
    """
    Map OpenAI-style audio model ids to a faster-whisper model name.

    This mirrors the behavior of the REST audio endpoint's
    `_map_openai_audio_model_to_whisper` helper so that Jobs use the same
    model mapping semantics (e.g., 'whisper-1' -> 'large-v3'). Unknown
    values fall back to 'large-v3'.
    """
    default_model = "large-v3"
    if not model:
        return default_model

    raw = str(model).strip()
    if not raw:
        return default_model

    m = raw.lower()
    valid_sizes = _valid_whisper_model_sizes_for_jobs()
    valid_sizes_lower = {s.lower() for s in valid_sizes}
    if not valid_sizes_lower:
        valid_sizes_lower = {
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large-v1",
            "large-v2",
            "large-v3",
            "large",
        }

    # Pass through known internal sizes and HF ids
    if raw in valid_sizes or m in valid_sizes or "/" in raw:
        return raw

    # OpenAI-compatible aliases
    if m == "whisper-1":
        return default_model
    if m in {"whisper-large-v3-turbo", "whisper-large-v3-turbo-ct2", "large-v3-turbo"}:
        return "deepdml/faster-whisper-large-v3-turbo-ct2"
    if m.startswith("whisper-") and m.endswith("-ct2"):
        ct2_tail = m[len("whisper-"):-4]
        if ct2_tail in valid_sizes_lower:
            return ct2_tail

    # Fallback to default
    return default_model


def _normalize_hotwords(hotwords: Optional[Sequence[str] | str]) -> Optional[list[str]]:
    """Normalize hotwords from either a list/sequence or a JSON/CSV string."""
    if hotwords is None:
        return None
    if isinstance(hotwords, str):
        raw = hotwords.strip()
        if not raw:
            return None
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    out = [str(x).strip() for x in parsed if str(x).strip()]
                    return out or None
            except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
                pass
        out = [part.strip() for part in raw.split(",") if part.strip()]
        return out or None
    out = [str(x).strip() for x in hotwords if str(x).strip()]
    return out or None


def _hotwords_cache_suffix(hotwords: Optional[Sequence[str] | str]) -> str:
    """
    Return a short, stable cache-key suffix derived from hotwords.

    When hotwords are provided, caching must differentiate between different
    hotword sets to avoid reusing an incompatible transcript.
    """
    hotwords_norm = _normalize_hotwords(hotwords)
    if not hotwords_norm:
        return ""
    canonical = sorted({str(x).strip() for x in hotwords_norm if str(x).strip()})
    if not canonical:
        return ""
    raw = json.dumps(canonical, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"-hotwords-{digest}"


def run_stt_batch_via_registry(
    audio_file_path: str,
    transcription_model: str,
    *,
    vad_filter: bool = False,
    selected_source_lang: str = "en",
    hotwords: Optional[Sequence[str] | str] = None,
    duration_seconds: Optional[float] = None,
    base_dir: Optional[Path] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    """
    Run batch STT via the shared provider registry and return a normalized artifact.

    This helper centralizes provider/model resolution for ingestion-style flows.
    For Whisper-family models it delegates to `re_generate_transcription` to
    preserve existing caching behaviour; for other providers it uses the
    adapter-based `transcribe_batch` implementation.

    Hotwords are passed through to providers that support them (for example
    VibeVoice-ASR) and ignored elsewhere.
    """
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (  # type: ignore
        get_stt_provider_registry,
    )

    registry = get_stt_provider_registry()
    provider, provider_model, _ = registry.resolve_provider_for_model(transcription_model or "")
    adapter = registry.get_adapter(provider)
    hotwords_norm = _normalize_hotwords(hotwords)

    # Whisper-family models: reuse the canonical regeneration helper so that
    # transcript cache files and error semantics remain unchanged.
    if provider == "faster-whisper":
        _check_cancel(cancel_check, label="stt batch")
        _, segments = re_generate_transcription(
            audio_file_path,
            transcription_model,
            vad_filter,
            selected_source_lang=selected_source_lang,
            base_dir=base_dir,
            cancel_check=cancel_check,
        )
        if segments is None:
            raise RuntimeError("STT transcription failed; no segments produced")

        text = " ".join(
            str(seg.get("Text", "")).strip()
            for seg in segments
            if isinstance(seg, dict)
        )
        return to_normalized_stt_artifact(
            text=text,
            segments=segments,
            language=selected_source_lang,
            provider=provider,
            model=transcription_model or provider_model,
            duration_seconds=duration_seconds,
        )

    # Non-Whisper providers: use adapter transcribe_batch directly.
    _check_cancel(cancel_check, label="stt batch")
    artifact = adapter.transcribe_batch(
        audio_file_path,
        model=transcription_model or provider_model,
        language=selected_source_lang,
        task="transcribe",
        word_timestamps=False,
        prompt=None,
        hotwords=hotwords_norm,
        base_dir=base_dir,
        cancel_check=cancel_check,
    )

    # Ensure duration_ms is set when we know duration.
    if duration_seconds is not None:
        try:
            duration_ms = round(max(float(duration_seconds), 0.0) * 1000)
            usage = artifact.setdefault("usage", {})
            if usage.get("duration_ms") is None:
                usage["duration_ms"] = duration_ms
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
            logging.debug(f"Failed to set duration_ms in artifact: {e}")

    return artifact


def run_stt_job_via_registry(
    wav_path: str,
    model: Optional[str],
    language: Optional[str],
    hotwords: Optional[Sequence[str] | str] = None,
    base_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Run STT for Jobs via the shared provider registry and return a normalized artifact.

    This helper is intended for worker-style flows (audio_jobs_worker,
    audio_transcribe_gpu_worker). It mirrors the REST endpoint's behavior for
    Whisper-family models by reusing the same OpenAI-style model mapping
    semantics (e.g., 'whisper-1' -> 'large-v3') while delegating to provider
    adapters for non-Whisper providers.

    Hotwords are passed through to providers that support them (for example
    VibeVoice-ASR) and ignored elsewhere.
    """
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (  # type: ignore
        get_stt_provider_registry,
    )

    registry = get_stt_provider_registry()
    # When no model is provided, resolve the config default via the registry.
    requested_model = (model or "").strip()
    provider, provider_model, _ = registry.resolve_provider_for_model(requested_model)
    adapter = registry.get_adapter(provider)
    hotwords_norm = _normalize_hotwords(hotwords)

    # Whisper-family models: reuse the OpenAI-compatible alias mapping so
    # that 'whisper-1' and similar identifiers resolve consistently across
    # REST, ingestion, and Jobs.
    if provider == "faster-whisper":
        whisper_model_name = _map_openai_audio_model_to_whisper_for_jobs(requested_model or "whisper-1")
        selected_lang = language or None
        artifact = adapter.transcribe_batch(
            wav_path,
            model=whisper_model_name,
            language=selected_lang,
            task="transcribe",
            word_timestamps=False,
            prompt=None,
            hotwords=hotwords_norm,
            base_dir=base_dir,
        )
        return artifact

    # Non-Whisper providers: use adapter transcribe_batch directly with the
    # resolved provider-specific model identifier.
    artifact = adapter.transcribe_batch(
        wav_path,
        model=requested_model or provider_model,
        language=language or None,
        task="transcribe",
        word_timestamps=False,
        prompt=None,
        hotwords=hotwords_norm,
        base_dir=base_dir,
    )
    return artifact


##########################################################
#
# Qwen2-Audio-related Functions

# Load Qwen2Audio (lazy load or load once at startup)
qwen_processor = None
qwen_model = None

def load_qwen2audio():
    """
    Loads the Qwen2Audio model and processor.

    This function implements lazy loading: the model and processor are loaded
    only on the first call and then cached in global variables for subsequent
    calls. It uses "Qwen/Qwen2-Audio-7B-Instruct" from Hugging Face Hub.

    Returns:
        A tuple `(processor, model)`:
        - `processor`: The `AutoProcessor` for Qwen2Audio.
        - `model`: The `Qwen2AudioForConditionalGeneration` model.

    Raises:
        RuntimeError: When Qwen2Audio is explicitly disabled or not configured
            in `[STT-Settings].qwen2audio_enabled`. In this case the error
            message is a sentinel string starting with
            "[Transcription error] Qwen2Audio is disabled or not configured"
            so higher-level helpers (e.g. `transcribe_audio`) can convert it
            into a consistent error sentinel for REST and speech-chat flows.
        ImportError: If `transformers` library is not installed.
        Exception: Propagates errors from `from_pretrained` if model downloading
            or loading fails (e.g., network issues, insufficient memory).
    """
    global qwen_processor, qwen_model
    if qwen_processor is None or qwen_model is None:
        torch_mod = _get_torch(allow_import=True)
        if torch_mod is None:
            raise RuntimeError(
                f"[Transcription error] Qwen2Audio unavailable because torch failed to import: {_TORCH_IMPORT_ERROR}"
            )
        qwen2audio_classes = _get_qwen2audio_classes()
        if qwen2audio_classes is None:
            raise RuntimeError(
                "[Transcription error] Qwen2Audio unavailable because transformers failed to import: "
                f"{_QWEN2AUDIO_IMPORT_ERROR}"
            )
        auto_processor_cls, qwen2audio_model_cls = qwen2audio_classes
        # Gate heavy Qwen2Audio loading behind config so typical installs
        # do not attempt to download/initialize this large model unless
        # explicitly enabled.
        cfg = load_and_log_configs() or {}
        stt_cfg = cfg.get("STT-Settings") or {}
        enabled_raw = stt_cfg.get("qwen2audio_enabled")
        if enabled_raw is None or not _to_bool(enabled_raw):
            logging.warning(
                "Qwen2Audio requested but STT-Settings.qwen2audio_enabled is not set or false; "
                "treating Qwen2Audio as disabled."
            )
            raise RuntimeError("[Transcription error] Qwen2Audio is disabled or not configured")

        configured_model_id = stt_cfg.get("qwen2audio_model_id", "Qwen/Qwen2-Audio-7B-Instruct")
        try:
            model_id = _normalize_qwen2audio_model_identifier(str(configured_model_id))
        except ValueError as exc:
            logging.warning(f"Rejected unsafe Qwen2Audio model identifier '{configured_model_id}': {exc}")
            raise RuntimeError("[Transcription error] Invalid Qwen2Audio model identifier") from exc
        revision = stt_cfg.get("qwen2audio_revision") or os.getenv("QWEN2AUDIO_REVISION")
        logging.info(f"Loading Qwen2Audio model: {model_id}")

        qwen_processor = auto_processor_cls.from_pretrained(model_id, revision=revision)  # nosec B615
        qwen_model = qwen2audio_model_cls.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch_mod.float16,
            device_map="auto"
        )  # nosec B615
    return qwen_processor, qwen_model

def transcribe_with_qwen2audio(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """
    Transcribes an audio waveform using the Qwen2Audio model.

    This function takes a raw audio NumPy array, processes it with the
    Qwen2Audio processor, and generates a transcription using the model's
    ASR capabilities. It uses a specific prompt structure required by
    Qwen2Audio for transcription tasks.

    Args:
        audio: A NumPy array representing the raw audio waveform (float32).
        sample_rate: The sample rate of the input `audio` in Hz.

    Returns:
        The transcribed text as a string. Returns an empty string or raises
        an exception if transcription fails; error sentinel strings with the
        "[Transcription error]" prefix are produced by higher-level helpers
        such as `transcribe_audio`, not by this function directly.

    Raises:
        RuntimeError: May propagate the sentinel-style `RuntimeError` raised by
            `load_qwen2audio` when Qwen2Audio is disabled or misconfigured.
        Exception: Any other error encountered while preparing model inputs
            or during `model.generate`.
    """
    processor, model = load_qwen2audio()

    # We build a prompt that includes <|audio_bos|><|AUDIO|><|audio_eos|> token(s)
    # The simplest approach is "User: <|AUDIO|>"
    # But Qwen2Audio also uses special tokens <|audio_bos|> and <|audio_eos|>.
    prompt_text = "System: You are a transcription model.\nUser: <|audio_bos|><|AUDIO|><|audio_eos|>\nAssistant:"

    inputs = processor(
        text=prompt_text,
        audios=audio,
        return_tensors="pt",
        sampling_rate=sample_rate
    )
    device = model.device
    torch_mod = _get_torch(allow_import=True)
    if torch_mod is None:
        raise RuntimeError(
            f"[Transcription error] Qwen2Audio unavailable because torch failed to import: {_TORCH_IMPORT_ERROR}"
        )
    for k, v in inputs.items():
        if isinstance(v, torch_mod.Tensor):
            inputs[k] = v.to(device)

    with torch_mod.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    # The raw output has prompt + transcription + possibly more text
    transcription = processor.decode(generated_ids[0], skip_special_tokens=True)

    # Post-process transcription
    # Qwen2Audio might produce additional text.
    # Typically you look for the part after "Assistant:"
    # or remove your system prompt if it appears in the output.
    # A quick approach:
    if "Assistant:" in transcription:
        # e.g. "System: ... User: <|AUDIO|>\nAssistant: Hello here's your text"
        transcription = transcription.split("Assistant:")[-1].strip()

    return transcription

#
# End of Qwen2-Audio-related Functions
##########################################################


##########################################################
#
# Faster Whisper related functions
whisper_model_instance = None
config = load_and_log_configs() or {}


def _resolve_processing_choice(config_data: dict[str, Any] | None) -> str:
    """Resolve the STT processing device, allowing a runtime env override."""
    env_override = str(os.getenv("PROCESSING_CHOICE", "")).strip().lower()
    if env_override:
        return env_override
    configured = str((config_data or {}).get("processing_choice", "cpu")).strip().lower()
    return configured or "cpu"


processing_choice = _resolve_processing_choice(config)
total_thread_count = multiprocessing.cpu_count()

# Model download status tracking
model_download_status = {}

def check_model_exists(model_name: str) -> bool:
    """
    Check if a Whisper model is already downloaded.

    Args:
        model_name: Name of the model to check

    Returns:
        True if model exists locally, False otherwise
    """
    if not model_name:
        return False

    # Resolve the default download root once as an absolute, normalized path.
    # All model paths must remain within this directory.
    default_root_path = WHISPER_MODEL_BASE_DIR.resolve(strict=False)

    try:
        normalized = _normalize_whisper_model_identifier(
            model_name,
            base_dir=default_root_path,
        )
    except ValueError as exc:
        logging.warning(f"Rejected unsafe whisper model identifier '{model_name}': {exc}")
        return False

    normalized_path = Path(normalized)
    if normalized_path.is_absolute():
        # The normalized path has already been resolved and validated to
        # reside under default_root_path by _normalize_whisper_model_identifier.
        # The path has already been validated; just check existence.
        return normalized_path.exists()

    # Check in default download directory for relative identifiers
    model_path = default_root_path / normalized
    if model_path.is_dir():
        return True

    # Check if it's a Hub ID that might be cached under our managed root.
    if _is_hf_model_id(normalized):
        # Convert Hub ID to potential cache path
        cache_name = normalized.replace('/', '_')
        cache_path = default_root_path / cache_name
        if cache_path.is_dir():
            return True

    # Do not probe global HuggingFace cache directories based on user-controlled
    # identifiers; if the model is not present under the configured root, treat
    # it as absent and allow the normal download mechanisms to handle it.
    return False

def set_model_download_status(model_name: str, status: str, message: str):
    """
    Set the download status for a model.

    Args:
        model_name: Name of the model
        status: Status of the download ('checking', 'downloading', 'completed', 'error')
        message: Human-readable status message
    """
    global model_download_status
    model_download_status[model_name] = {
        'status': status,
        'message': message,
        'timestamp': time.time()
    }
    logging.info(f"Model download status for {model_name}: {status} - {message}")

def get_model_download_status(model_name: str) -> Optional[dict[str, Any]]:
    """
    Get the current download status for a model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with status information or None if no status
    """
    return model_download_status.get(model_name)

class WhisperModel:
    """
    Lazy wrapper for `faster_whisper.WhisperModel` to manage model loading.

    This class preserves the public `WhisperModel` API used by the rest of the
    codebase while deferring the actual `faster_whisper` import until an
    instance is created. It keeps the existing custom model path resolution and
    default download-root behavior.

    Attributes:
        default_download_root (str): The default directory path where models
            will be downloaded or looked for if not found elsewhere.
        valid_model_sizes (List[str]): A list of recognized standard model size
            names and some known community model identifiers.
    """
    default_download_root = str(WHISPER_MODEL_BASE_DIR)

    valid_model_sizes = [
        "tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium",
        "large-v1", "large-v2", "large-v3", "large", "distil-large-v2", "distil-medium.en",
        "distil-small.en", "distil-large-v3", "deepdml/faster-distil-whisper-large-v3.5", "deepdml/faster-whisper-large-v3-turbo-ct2",
        "nyrahealth/faster_CrisperWhisper"
    ]

    def __init__(
        self,
        model_size_or_path: str,
        device: str = processing_choice,
        device_index: Union[int, list[int]] = 0,
        compute_type: str = "default",
        cpu_threads: int = 0,#total_thread_count, FIXME - I think this should be 0
        num_workers: int = 1,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        files: Optional[dict[str, Any]] = None,
        **model_kwargs: Any
    ):
        """
        Initializes the custom WhisperModel.

        Determines if `model_size_or_path` is a Hugging Face Hub ID, an
        existing local path, or a standard model name. It then calls the
        parent `faster_whisper.WhisperModel` initializer with the resolved
        identifier and specified `download_root`.

        Args:
            model_size_or_path: Identifier for the model. Can be:
                - A standard model size name (e.g., "large-v3", "tiny.en").
                - A path to a local model directory.
                - A Hugging Face Hub model ID (e.g., "openai/whisper-large-v3").
            device: Device to load the model on ("cpu", "cuda", "auto").
            device_index: Index of the device(s) to use.
            compute_type: Type of computation to use (e.g., "float16", "int8").
            cpu_threads: Number of CPU threads to use for inference.
                Set to 0 for faster-whisper to auto-detect.
            num_workers: Number of workers for parallel transcription.
            download_root: Path to the directory for downloading/caching models.
                If None, uses `WhisperModel.default_download_root`. The path must
                resolve under `WHISPER_MODEL_BASE_DIR`.
            local_files_only: If True, only look for local files and do not
                attempt to download.
            files: Optional dictionary of specific files to use for the model,
                   as per faster-whisper's `OriginalWhisperModel`.
            **model_kwargs: Additional keyword arguments passed to the
                `faster_whisper.WhisperModel` constructor.

        Raises:
            ValueError: If the model identifier is invalid, cannot be resolved,
                or if `faster_whisper.WhisperModel` initialization fails.
            RuntimeError: For other unexpected errors during model loading.
        """
        original_model_cls = _get_original_whisper_model()
        if original_model_cls is None:
            raise RuntimeError(
                "faster_whisper unavailable because import failed: "
                f"{_ORIGINAL_WHISPER_MODEL_IMPORT_ERROR}"
            )

        download_root_path = _resolve_whisper_download_root(
            download_root or self.default_download_root
        )

        try:
            resolved_identifier = _normalize_whisper_model_identifier(
                model_size_or_path,
                base_dir=download_root_path,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        resolved_path = Path(resolved_identifier)
        is_local_path = resolved_path.is_absolute() and resolve_safe_local_path(
            resolved_path,
            download_root_path,
        )

        if is_local_path:
            # It's a local path that exists under the allowed model root.
            logging.info(f"Treating '{resolved_identifier}' as an existing local path.")
            resolved_identifier = str(resolved_path)
        elif _is_hf_model_id(resolved_identifier):
            # Assume it's a Hub ID - pass it directly to faster-whisper.
            # faster-whisper will handle downloading it (potentially respecting download_root if configured).
            logging.info(f"Treating '{resolved_identifier}' as a Hugging Face Hub ID.")
        else:
            # Assume it's a standard model size name (e.g., "large-v3").
            # Let faster-whisper handle finding/downloading this standard model.
            logging.info(f"Treating '{resolved_identifier}' as a standard model size name.")
            local_path = _check_standard_model_under_download_root(
                resolved_identifier,
                download_root_path,
            )
            if local_path is not None:
                resolved_identifier = str(local_path)


        # --- Pass the determined identifier and other args to the parent ---
        logging.info(
             f"Initializing faster-whisper with: model='{resolved_identifier}', "
             f"device='{device}', compute_type='{compute_type}', "
             f"download_root='{download_root_path}', local_files_only={local_files_only}"
        )

        try:
            self._delegate = original_model_cls(
                model_size_or_path=resolved_identifier, # Use the corrected identifier
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
                download_root=str(download_root_path), # Pass your custom root
                local_files_only=local_files_only,
                **model_kwargs
            )
            self.model_identifier = resolved_identifier # Store for reference if needed
            logging.info(f"Successfully initialized WhisperModel: {resolved_identifier}")

        except ValueError as e:
            # Error during faster-whisper init (e.g., invalid model, download failed)
            logging.error(f"Failed to initialize faster_whisper.WhisperModel with '{resolved_identifier}': {e}", exc_info=True)
            # Provide a more specific error message based on the likely cause
            if "Invalid model size" in str(e) or "could not be found" in str(e):
                 raise ValueError(f"The model identifier '{resolved_identifier}' is invalid or could not be loaded/downloaded. Check the name/path and ensure it's accessible.") from e
            else:
                 raise ValueError(f"Error initializing model '{resolved_identifier}': {e}") from e
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
             # Catch other unexpected errors
             logging.error(f"An unexpected error occurred during faster_whisper.WhisperModel initialization with '{resolved_identifier}': {e}", exc_info=True)
             raise RuntimeError(f"Unexpected error loading model: {resolved_identifier} - {e}") from e

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

# Model unloading functions
def unload_whisper_model():
    """
    Unloads the global faster-whisper model instance and triggers garbage collection.

    This function is intended to free up resources, particularly GPU memory,
    used by the loaded Whisper model. It deletes the reference to the global
    `whisper_model_instance` (if it exists and was set by `get_whisper_model` or
    directly) and also clears the `whisper_model_cache`.

    Note:
        If `whisper_model_instance` was not the sole reference to the model object
        (e.g., if it's also in `whisper_model_cache` and that cache is used),
        deleting it alone might not free memory until the cache is also cleared
        or the Python garbage collector reclaims the object. This function now
        explicitly clears the cache.
    """
    global whisper_model_instance
    whisper_model_instance = None  # kept for backward compat
    with whisper_model_cache_lock:
        whisper_model_cache.clear()
    gc.collect()


def unload_all_transcription_models():
    """
    Unload all transcription models (Whisper, Qwen, Nemo) to free memory.
    """
    # Unload Whisper models
    unload_whisper_model()

    # Unload Qwen2Audio models
    global qwen_processor, qwen_model
    if qwen_processor is not None:
        del qwen_processor
        qwen_processor = None
    if qwen_model is not None:
        del qwen_model
        qwen_model = None

    # Unload Nemo models
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            unload_nemo_models,
        )
        unload_nemo_models()
    except ImportError:
        pass  # Nemo module not available

    # Force garbage collection
    gc.collect()

    # Clear GPU cache if available
    torch_mod = _get_torch(allow_import=False)
    if _torch_cuda_available(allow_import=False) and torch_mod is not None:
        torch_mod.cuda.empty_cache()

    logging.info("Unloaded all transcription models from memory")

whisper_model_cache = {}
whisper_model_cache_lock = threading.Lock()

def get_whisper_model(model_name, device, check_download_status=False):
    """
    Retrieves or initializes a `WhisperModel` instance, using a cache.

    This function checks a cache for an existing model instance matching the
    `model_name`, `device`, and a determined `compute_type`. If not found,
    it initializes a new `WhisperModel` instance, stores it in the cache,
    and returns it.

    The `compute_type` is resolved as follows:
      - When `[STT-Settings].whisper_compute_type` is set to a non-empty value
        other than "auto", that value is passed through to the underlying
        faster-whisper constructor (for example: "float16", "int8",
        "int8_float16").
      - Otherwise, the compute type defaults to "float16" when `device`
        contains "cuda" and "int8" for CPU devices.

    Args:
        model_name: The name or path of the Whisper model (e.g., "base.en",
            "/path/to/model", "openai/whisper-large-v3").
        device: The device to load the model on ("cpu", "cuda").
        check_download_status: If True, check if model needs downloading and return status.

    Returns:
        A `WhisperModel` instance, or a tuple (None, status_dict) if check_download_status
        is True and model needs downloading.

    Raises:
        ValueError: If `WhisperModel` initialization fails (e.g., invalid model name).
        RuntimeError: For other unexpected errors during model loading.
    """
    # Optional override from STT-Settings; when unset or "auto", fall back to
    # the prior device-based heuristic.
    if WHISPER_COMPUTE_TYPE_OVERRIDE and WHISPER_COMPUTE_TYPE_OVERRIDE != "auto":
        compute_type = WHISPER_COMPUTE_TYPE_OVERRIDE
    else:
        compute_type = "float16" if "cuda" in device else "int8"
    try:
        normalized_model_name = _normalize_whisper_model_identifier(
            model_name,
            base_dir=WHISPER_MODEL_BASE_DIR,
        )
    except ValueError as exc:
        logging.error(f"Invalid whisper model identifier '{model_name}': {exc}")
        raise ValueError(str(exc)) from exc

    cache_key = (normalized_model_name, device, compute_type)

    with whisper_model_cache_lock:
        # If checking download status and model not in cache
        if check_download_status and cache_key not in whisper_model_cache:
            if not check_model_exists(normalized_model_name):
                return None, {
                    'status': 'model_downloading',
                    'message': (
                        f'Model {normalized_model_name} is not available locally and will be '
                        'downloaded on first use. This may take several minutes '
                        'depending on your internet connection.'
                    ),
                    'model': normalized_model_name
                }

        if cache_key not in whisper_model_cache:
            logging.info(f"Cache miss. Initializing WhisperModel for key: {cache_key}")
            try:
                # This now calls the *corrected* WhisperModel.__init__
                instance = WhisperModel(
                    model_size_or_path=normalized_model_name,
                    device=device,
                    compute_type=compute_type
                )
                whisper_model_cache[cache_key] = instance
            except (ValueError, RuntimeError) as e:
                # Check if the error is related to CUDA not being available
                if "cuda" in device.lower() and ("CUDA" in str(e) or "cuda" in str(e).lower()):
                    logging.warning(f"CUDA initialization failed for {normalized_model_name}: {e}. Falling back to CPU.")
                    # Try again with CPU
                    cpu_compute_type = "int8"
                    cpu_cache_key = (normalized_model_name, "cpu", cpu_compute_type)

                    if cpu_cache_key not in whisper_model_cache:
                        try:
                            instance = WhisperModel(
                                model_size_or_path=normalized_model_name,
                                device="cpu",
                                compute_type=cpu_compute_type
                            )
                            whisper_model_cache[cpu_cache_key] = instance
                            # Also cache it under the original key to avoid retrying CUDA
                            whisper_model_cache[cache_key] = instance
                            logging.info("Successfully initialized WhisperModel on CPU as fallback")
                            return instance
                        except (ValueError, RuntimeError) as cpu_e:
                            logging.error(f"Failed to initialize WhisperModel on CPU as well: {cpu_e}")
                            raise RuntimeError(f"Failed to initialize model on both CUDA and CPU: {cpu_e}") from cpu_e
                    else:
                        # Use existing CPU instance
                        instance = whisper_model_cache[cpu_cache_key]
                        whisper_model_cache[cache_key] = instance  # Cache under original key too
                        return instance
                else:
                    logging.error(f"Fatal error creating whisper model instance for key {cache_key}: {e}")
                    raise  # Re-raise the exception
        else:
            logging.debug(f"Cache hit. Reusing existing WhisperModel instance for key: {cache_key}")

        return whisper_model_cache[cache_key]


# Transcribe .wav into .segments.json
#DEBUG
#@profile
# FIXME - I feel like the `vad_filter` should be enabled by default....
@timeit
def format_time(total_seconds: float) -> str:
    """
    Convert a float number of seconds into HH:MM:SS format.
    E.g., 123.45 -> '00:02:03'
    """
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def parse_transcription_model(model_name: str) -> tuple:
    """
    Parse model name to extract provider, model, and variant.

    Examples:
    - "parakeet-mlx" -> ("parakeet", "parakeet", "mlx")
    - "parakeet-onnx" -> ("parakeet", "parakeet", "onnx")
    - "parakeet-cuda" -> ("parakeet", "parakeet", "cuda")
    - "parakeet-standard" -> ("parakeet", "parakeet", "standard")
    - "nemo-canary-1b" -> ("canary", "canary", "standard")
    - "nemo-parakeet-tdt-1.1b" -> ("parakeet", "parakeet", "standard")
    - "whisper-large-v3" -> ("whisper", "large-v3", None)
    - "distil-whisper-large-v3" -> ("whisper", "distil-large-v3", None)

    Returns:
        Tuple of (provider, model, variant)
    """
    model_name = (model_name or "").strip()
    model_lower = model_name.lower()

    # Check for Parakeet models with variants
    if "parakeet" in model_lower:
        if model_lower.endswith("-mlx"):
            return ("parakeet", "parakeet", "mlx")
        elif model_lower.endswith("-onnx"):
            return ("parakeet", "parakeet", "onnx")
        elif model_lower.endswith("-cuda"):
            return ("parakeet", "parakeet", "cuda")
        elif model_lower.endswith("-standard") or "nemo-parakeet" in model_lower:
            return ("parakeet", "parakeet", "standard")
        else:
            # Use config default when no variant is specified.
            try:
                stt_cfg = get_stt_config()
            except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
                stt_cfg = {}
            variant = str(stt_cfg.get("nemo_model_variant", "standard")).strip().lower()
            if variant not in {"standard", "onnx", "mlx", "cuda"}:
                variant = "standard"
            return ("parakeet", "parakeet", variant)

    # Check for Canary models
    elif "canary" in model_lower:
        return ("canary", "canary", "standard")

    # Check for Qwen2Audio models
    elif "qwen2audio" in model_lower or "qwen2-audio" in model_lower:
        return ("qwen2audio", model_name, None)

    # Check for VibeVoice-ASR models
    elif "vibevoice" in model_lower:
        # Treat bare aliases as the configured default model id.
        if model_lower in {"vibevoice", "vibevoice-asr", "vibevoice_asr"}:
            try:
                stt_cfg = get_stt_config() or {}
            except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
                stt_cfg = {}
            model_id = str(stt_cfg.get("vibevoice_model_id", "microsoft/VibeVoice-ASR")).strip()
            return ("vibevoice", model_id or "microsoft/VibeVoice-ASR", None)
        return ("vibevoice", model_name, None)

    # Check for Qwen3-ASR models
    elif "qwen3" in model_lower and "asr" in model_lower:
        # Get config path and derive the appropriate model path based on size
        try:
            stt_cfg = get_stt_config() or {}
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
            stt_cfg = {}
        base_path = str(stt_cfg.get("qwen3_asr_model_path", "./models/qwen3_asr/1.7B")).strip()
        if not base_path:
            base_path = "./models/qwen3_asr/1.7B"
        # If user requests 0.6B, derive path by replacing 1.7B with 0.6B in the configured path
        if "0.6b" in model_lower:
            # Try to derive 0.6B path from the configured 1.7B path
            if "1.7B" in base_path or "1.7b" in base_path:
                model_path = base_path.replace("1.7B", "0.6B").replace("1.7b", "0.6b")
            else:
                # If the path doesn't contain 1.7B, append 0.6B to parent dir
                model_path = str(Path(base_path).parent / "0.6B")
        else:
            # Default to configured path (typically 1.7B)
            model_path = base_path
        return ("qwen3-asr", model_path, None)

    # Default to whisper for all other models
    else:
        # This includes whisper-*, distil-whisper-*, deepdml/*, etc.
        whisper_prefix = "whisper-"
        distil_prefix = "distil-whisper-"
        whisper_sizes = {
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large-v1",
            "large-v2",
            "large-v3",
            "large-v3-turbo",
            "large",
            "turbo",
        }
        distil_sizes = {
            "large-v2",
            "large-v3",
            "large-v3.5",
            "medium.en",
            "small.en",
        }

        if model_lower == "whisper-1":
            return ("whisper", "large-v3", None)
        if model_lower in {"whisper-large-v3-turbo", "whisper-large-v3-turbo-ct2"}:
            return ("whisper", "deepdml/faster-whisper-large-v3-turbo-ct2", None)

        if model_lower.startswith(distil_prefix):
            tail = model_lower[len(distil_prefix):]
            if tail in distil_sizes:
                return ("whisper", f"distil-{tail}", None)

        if model_lower.startswith(whisper_prefix):
            tail = model_lower[len(whisper_prefix):]
            if tail.endswith("-ct2"):
                ct2_tail = tail[:-4]
                if ct2_tail in whisper_sizes:
                    return ("whisper", ct2_tail, None)
            if tail in whisper_sizes:
                return ("whisper", tail, None)

        return ("whisper", model_name, None)

def create_segments_from_text(text: str, audio_duration: float = None, segmentation: str = "sentence") -> list:
    """
    Convert plain text to whisper-compatible segment format with sentence or word-level segmentation.

    Args:
        text: The transcribed text
        audio_duration: Optional duration of the audio in seconds
        segmentation: Segmentation level - "sentence" (default), "word", or "full"

    Returns:
        List of segment dictionaries compatible with whisper format
    """
    import re

    if not text:
        return []

    text = text.strip()
    segments = []

    if segmentation == "sentence":
        # Split by sentence-ending punctuation, keeping the punctuation
        # This regex splits on .!? followed by space or end of string, keeping the delimiter
        sentence_pattern = r'(?<=[.!?])\s+|(?<=[.!?])$'
        sentences = re.split(sentence_pattern, text)
        # Remove empty strings
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            sentences = [text]  # Fallback if no sentence boundaries found

        # Estimate time distribution across sentences
        if audio_duration:
            # Calculate time per character for even distribution
            total_chars = sum(len(s) for s in sentences)
            time_per_char = audio_duration / total_chars if total_chars > 0 else 0

            current_time = 0.0
            for sentence in sentences:
                sentence_duration = len(sentence) * time_per_char
                end_time = current_time + sentence_duration

                segments.append({
                    "start_seconds": current_time,
                    "end_seconds": end_time,
                    "text": sentence,
                    "Text": sentence,  # Some code expects "Text" key
                    "start": format_time(current_time),
                    "end": format_time(end_time),
                    "Time_Start": current_time,
                    "Time_End": end_time
                })

                current_time = end_time
        else:
            # No duration info, just create segments without timing
            for _i, sentence in enumerate(sentences):
                segments.append({
                    "start_seconds": 0.0,
                    "end_seconds": 0.0,
                    "text": sentence,
                    "Text": sentence,
                    "start": "00:00:00",
                    "end": "00:00:00",
                    "Time_Start": 0.0,
                    "Time_End": 0.0
                })

    elif segmentation == "word":
        # Split by whitespace to get words
        words = text.split()

        if audio_duration and words:
            # Distribute time evenly across words
            time_per_word = audio_duration / len(words)

            for i, word in enumerate(words):
                start_time = i * time_per_word
                end_time = (i + 1) * time_per_word

                segments.append({
                    "start_seconds": start_time,
                    "end_seconds": end_time,
                    "text": word,
                    "Text": word,
                    "start": format_time(start_time),
                    "end": format_time(end_time),
                    "Time_Start": start_time,
                    "Time_End": end_time
                })
        else:
            # No duration info
            for word in words:
                segments.append({
                    "start_seconds": 0.0,
                    "end_seconds": 0.0,
                    "text": word,
                    "Text": word,
                    "start": "00:00:00",
                    "end": "00:00:00",
                    "Time_Start": 0.0,
                    "Time_End": 0.0
                })

    else:  # "full" or any other value
        # Single segment with full text
        segments = [{
            "start_seconds": 0.0,
            "end_seconds": audio_duration if audio_duration else 0.0,
            "text": text,
            "Text": text,
            "start": "00:00:00",
            "end": format_time(audio_duration) if audio_duration else "00:00:00",
            "Time_Start": 0.0,
            "Time_End": audio_duration if audio_duration else 0.0
        }]

    return segments


def create_segments_from_parakeet_mlx_artifact(
    artifact: Any,
    audio_duration: Optional[float] = None,
) -> list[dict[str, Any]]:
    """
    Convert a Parakeet MLX structured artifact into whisper-compatible segments.

    Expected artifact shape:
    {
      "text": str,
      "sentences": [
        {
          "text": str,
          "start": float,
          "end": float,
          "confidence": float,
          "tokens": [{"text": str, "start": float, "end": float, ...}]
        },
      ],
      "tokens": [...]
    }
    """
    if not isinstance(artifact, dict):
        text = str(artifact or "").strip()
        return create_segments_from_text(text, audio_duration, segmentation="sentence")

    text = str(artifact.get("text", "") or "").strip()
    raw_sentences = artifact.get("sentences")
    if not isinstance(raw_sentences, list):
        return create_segments_from_text(text, audio_duration, segmentation="sentence")

    segments: list[dict[str, Any]] = []
    for sentence in raw_sentences:
        if not isinstance(sentence, dict):
            continue
        sentence_text = str(sentence.get("text", "") or "").strip()
        if not sentence_text:
            continue

        sentence_start = _coerce_float(sentence.get("start"), 0.0)
        sentence_end = _coerce_float(sentence.get("end"), sentence_start)
        if sentence_end < sentence_start:
            sentence_end = sentence_start

        segment: dict[str, Any] = {
            "start_seconds": float(sentence_start),
            "end_seconds": float(sentence_end),
            "text": sentence_text,
            "Text": sentence_text,
            "start": format_time(float(sentence_start)),
            "end": format_time(float(sentence_end)),
            "Time_Start": float(sentence_start),
            "Time_End": float(sentence_end),
        }
        sentence_confidence = _coerce_float(sentence.get("confidence"))
        if sentence_confidence is not None:
            segment["confidence"] = float(sentence_confidence)

        raw_tokens = sentence.get("tokens")
        words: list[dict[str, Any]] = []
        if isinstance(raw_tokens, list):
            for token in raw_tokens:
                if not isinstance(token, dict):
                    continue
                token_text = str(token.get("text", "") or "").strip()
                token_start = _coerce_float(token.get("start"))
                token_end = _coerce_float(token.get("end"))
                if not token_text or token_start is None or token_end is None:
                    continue
                if token_end < token_start:
                    token_end = token_start
                words.append(
                    {
                        "start": float(token_start),
                        "end": float(token_end),
                        "word": token_text,
                    }
                )
        if words:
            segment["words"] = words

        segments.append(segment)

    if not segments:
        return create_segments_from_text(text, audio_duration, segmentation="sentence")
    return segments


def speech_to_text_parakeet(
    audio_file_path: str,
    variant: str = "standard",
    selected_source_lang: str = 'en',
    vad_filter: bool = False,
    base_dir: Optional[Path] = None,
) -> list:
    """
    Transcribe audio using Parakeet with specified variant.

    Args:
        audio_file_path: Path to the audio file
        variant: Parakeet variant ('standard', 'onnx', 'mlx', 'cuda')
        selected_source_lang: Language code (not used by Parakeet currently)
        vad_filter: VAD filter flag (not used by Parakeet currently)
        base_dir: Optional base directory used to validate local input paths.

    Returns:
        List of segments in whisper-compatible format
    """
    try:
        logging.info(f"Transcribing with Parakeet variant: {variant}")

        audio_path = _resolve_audio_input_path_for_provider(
            audio_file_path,
            base_dir=base_dir,
            label="Audio input path",
        )
        audio_file_path = str(audio_path)

        # Get audio duration for segment creation
        try:
            import librosa
            audio_duration = librosa.get_duration(path=audio_file_path)
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
            audio_duration = None
            logging.warning("Could not determine audio duration")

        # Route to appropriate Parakeet implementation
        if variant == "mlx":
            # Use MLX implementation (macOS only)
            try:
                from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
                    check_mlx_available,
                    transcribe_with_parakeet_mlx,
                )

                if not check_mlx_available():
                    logging.warning("MLX not available on this platform, falling back to standard Parakeet")
                    variant = "standard"
                else:
                    stt_cfg = get_stt_config() or {}
                    raw_chunk_duration = stt_cfg.get("mlx_chunk_duration")
                    raw_overlap_duration = stt_cfg.get("mlx_overlap_duration")
                    mlx_model_id = str(stt_cfg.get("mlx_model_id", "")).strip() or None
                    mlx_cache_dir = str(stt_cfg.get("mlx_cache_dir", "")).strip() or None
                    mlx_decoding_mode = str(stt_cfg.get("mlx_decoding_mode", "")).strip() or None
                    mlx_beam_size = _coerce_int(stt_cfg.get("mlx_beam_size"))
                    mlx_length_penalty = _coerce_float(stt_cfg.get("mlx_length_penalty"))
                    mlx_patience = _coerce_float(stt_cfg.get("mlx_patience"))
                    mlx_duration_reward = _coerce_float(stt_cfg.get("mlx_duration_reward"))
                    mlx_sentence_max_words = _coerce_int(stt_cfg.get("mlx_sentence_max_words"))
                    mlx_sentence_silence_gap = _coerce_float(stt_cfg.get("mlx_sentence_silence_gap"))
                    mlx_sentence_max_duration = _coerce_float(stt_cfg.get("mlx_sentence_max_duration"))

                    chunk_duration = _coerce_float(raw_chunk_duration)
                    if chunk_duration is None:
                        chunk_duration = _coerce_float(stt_cfg.get("buffered_chunk_duration"))
                    if chunk_duration is None:
                        chunk_duration = 30.0

                    overlap_duration = _coerce_float(raw_overlap_duration)
                    if overlap_duration is None:
                        overlap_duration = 5.0

                    mlx_result: Union[str, dict[str, Any]]
                    if chunk_duration <= 0:
                        mlx_result = transcribe_with_parakeet_mlx(
                            audio_file_path,
                            chunk_duration=None,
                            overlap_duration=15.0,
                            return_structured=True,
                            model_path=mlx_model_id,
                            cache_dir=mlx_cache_dir,
                            decoding_mode=mlx_decoding_mode,
                            beam_size=mlx_beam_size,
                            length_penalty=mlx_length_penalty,
                            patience=mlx_patience,
                            duration_reward=mlx_duration_reward,
                            sentence_max_words=mlx_sentence_max_words,
                            sentence_silence_gap=mlx_sentence_silence_gap,
                            sentence_max_duration=mlx_sentence_max_duration,
                        )
                    else:
                        use_buffered = audio_duration is None or audio_duration > chunk_duration
                        if use_buffered:
                            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
                                MergeAlgorithm,
                                transcribe_long_audio,
                            )

                            merge_algo = stt_cfg.get("buffered_merge_algo", "middle")
                            try:
                                MergeAlgorithm(merge_algo)
                            except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
                                merge_algo = "middle"

                            total_buffer = None
                            if "buffered_total_buffer" in stt_cfg:
                                total_buffer = _coerce_float(stt_cfg.get("buffered_total_buffer"))
                                if total_buffer is not None and (
                                    total_buffer <= chunk_duration or total_buffer >= 3.0 * chunk_duration
                                ):
                                    total_buffer = None
                            else:
                                total_buffer = chunk_duration + (2.0 * overlap_duration)
                                if total_buffer <= chunk_duration or total_buffer >= 3.0 * chunk_duration:
                                    total_buffer = None

                            mlx_result = transcribe_long_audio(
                                audio_file_path,
                                model_name="parakeet",
                                variant="mlx",
                                chunk_duration=chunk_duration,
                                total_buffer=total_buffer,
                                merge_algo=merge_algo,
                                return_structured=True,
                                model_path=mlx_model_id,
                                cache_dir=mlx_cache_dir,
                                decoding_mode=mlx_decoding_mode,
                                beam_size=mlx_beam_size,
                                length_penalty=mlx_length_penalty,
                                patience=mlx_patience,
                                duration_reward=mlx_duration_reward,
                                sentence_max_words=mlx_sentence_max_words,
                                sentence_silence_gap=mlx_sentence_silence_gap,
                                sentence_max_duration=mlx_sentence_max_duration,
                            )
                        else:
                            mlx_result = transcribe_with_parakeet_mlx(
                                audio_file_path,
                                chunk_duration=chunk_duration,
                                overlap_duration=overlap_duration,
                                return_structured=True,
                                model_path=mlx_model_id,
                                cache_dir=mlx_cache_dir,
                                decoding_mode=mlx_decoding_mode,
                                beam_size=mlx_beam_size,
                                length_penalty=mlx_length_penalty,
                                patience=mlx_patience,
                                duration_reward=mlx_duration_reward,
                                sentence_max_words=mlx_sentence_max_words,
                                sentence_silence_gap=mlx_sentence_silence_gap,
                                sentence_max_duration=mlx_sentence_max_duration,
                            )

                    if isinstance(mlx_result, str) and _looks_like_error_text(mlx_result):
                        raise STTTranscriptionError(mlx_result)

                    return create_segments_from_parakeet_mlx_artifact(mlx_result, audio_duration)

            except ImportError as e:
                logging.warning(f"Could not import Parakeet MLX: {e}. Falling back to standard.")
                variant = "standard"

        # For other variants or fallback, use Nemo implementation
        import librosa

        # Load audio data for Nemo implementation
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            transcribe_with_parakeet,
        )
        audio_data, sample_rate = librosa.load(audio_file_path, sr=16000, mono=True)

        # Transcribe with Parakeet
        text = transcribe_with_parakeet(audio_data, sample_rate, variant)

        # Convert to segment format with sentence-level segmentation
        return create_segments_from_text(text, audio_duration, segmentation="sentence")

    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Parakeet transcription failed: {e}")
        raise STTTranscriptionError(f"Parakeet transcription error: {str(e)}") from e

def speech_to_text_canary(
    audio_file_path: str,
    selected_source_lang: str = 'en',
    vad_filter: bool = False,
    base_dir: Optional[Path] = None,
) -> list:
    """
    Transcribe audio using Canary model.

    Args:
        audio_file_path: Path to the audio file
        selected_source_lang: Language code
        vad_filter: VAD filter flag (not used by Canary currently)
        base_dir: Optional base directory used to validate local input paths.

    Returns:
        List of segments in whisper-compatible format
    """
    try:
        logging.info("Transcribing with Canary model")

        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            transcribe_with_canary,
        )

        audio_path = _resolve_audio_input_path_for_provider(
            audio_file_path,
            base_dir=base_dir,
            label="Audio input path",
        )
        audio_file_path = str(audio_path)

        # Load audio data
        import librosa
        audio_data, sample_rate = librosa.load(audio_file_path, sr=16000, mono=True)

        # Get audio duration
        audio_duration = librosa.get_duration(y=audio_data, sr=sample_rate)

        # Transcribe with Canary
        text = transcribe_with_canary(audio_data, sample_rate, selected_source_lang)

        # Convert to segment format with sentence-level segmentation
        return create_segments_from_text(text, audio_duration, segmentation="sentence")

    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Canary transcription failed: {e}")
        raise RuntimeError(f"Canary transcription error: {str(e)}") from e

def speech_to_text_qwen2audio(
    audio_file_path: str,
    selected_source_lang: str = 'en',
    vad_filter: bool = False,
    base_dir: Optional[Path] = None,
) -> list:
    """
    Transcribe audio using Qwen2Audio model.

    Args:
        audio_file_path: Path to the audio file
        selected_source_lang: Language code (not used by Qwen2Audio currently)
        vad_filter: VAD filter flag (not used by Qwen2Audio currently)
        base_dir: Optional base directory used to validate local input paths.

    Returns:
        List of segments in whisper-compatible format
    """
    try:
        logging.info("Transcribing with Qwen2Audio model via speech_to_text_qwen2audio")

        audio_path = _resolve_audio_input_path_for_provider(
            audio_file_path,
            base_dir=base_dir,
            label="Audio input path",
        )
        audio_file_path = str(audio_path)

        # Validate provider availability/config before decoding audio so
        # disabled Qwen routes can fall back cleanly without decode errors.
        load_qwen2audio()

        # Load audio data
        import librosa
        audio_data, sample_rate = librosa.load(audio_file_path, sr=16000, mono=True)

        # Get audio duration
        audio_duration = librosa.get_duration(y=audio_data, sr=sample_rate)

        # Transcribe with Qwen2Audio
        text = transcribe_with_qwen2audio(audio_data, sample_rate)

        # Convert to segment format with sentence-level segmentation
        return create_segments_from_text(text, audio_duration, segmentation="sentence")

    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Qwen2Audio transcription failed: {e}")
        raise RuntimeError(f"Qwen2Audio transcription error: {str(e)}") from e

def speech_to_text(
    audio_input: Union[str, Path, np.ndarray],
    whisper_model: str = "distil-large-v3",
    selected_source_lang: str = "en",  # Changed order of parameters
    vad_filter: bool = False,
    diarize: bool = False,
    *,
    word_timestamps: bool = False,
    return_language: bool = False,
    persist_segments: Optional[bool] = None,
    cache_max_age_days: Optional[int] = None,
    cache_max_total_mb: Optional[float] = None,
    cache_max_files_per_source: Optional[int] = None,
    initial_prompt: Optional[str] = None,
    hotwords: Optional[Sequence[str] | str] = None,
    task: str = "transcribe",
    input_sample_rate: Optional[int] = None,
    base_dir: Optional[Path] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    include_metadata_header: bool = True,
) -> Union[list[dict[str, Any]], tuple[list[dict[str, Any]], Optional[str]]]:
    """
    Canonical file/segment-based speech-to-text helper.

    This function is the primary entry point for offline/media workflows that
    work with files (or occasionally in-memory waveforms). It loads the
    requested model/provider, performs transcription, and returns structured
    segments suitable for storage and downstream processing.

    Args:
        audio_input: Path to the WAV audio file to be transcribed, or a NumPy array
            of audio samples.
        whisper_model: Name or path of the faster-whisper model to use
            (e.g., 'distil-large-v3', 'base.en').
        selected_source_lang: Language code of the source audio (e.g., 'en', 'es').
            If `None`, the model will attempt to auto-detect the language.
            Defaults to 'en'.
        vad_filter: If True, applies Voice Activity Detection filter during
            transcription to potentially improve accuracy by filtering out non-speech
            segments.
        diarize: Placeholder for diarization flag. This parameter is not currently
            used within this function's transcription logic.
        hotwords: Optional hotword hints. Accepts a list/sequence or a JSON/CSV
            string. This is currently used by the VibeVoice-ASR provider and
            ignored by others.
        task: Whisper decoding task to perform. Valid values are "transcribe"
            (default, transcribe in source language) and "translate" (translate
            non-English speech to English, when supported by the underlying
            model/provider). For non-Whisper providers this value is ignored and
            transcription is always performed.
        input_sample_rate: Sample rate of the provided NumPy array input. Ignored when a
            file path is provided.
        base_dir: Optional base directory used to validate local file paths and
            anchor transcript cache output. When provided, audio file paths must
            resolve under this directory.

    Returns:
        By default, a list of segment dictionaries. Each dictionary contains:
        - "start_seconds" (float): Start time of the segment in seconds.
        - "end_seconds" (float): End time of the segment in seconds.
        - "Text" (str): The transcribed text of the segment.
        - Optional "words" (list): When word_timestamps=True, a list of
          {"start": float, "end": float, "word": str} entries per segment.
        When `include_metadata_header` is True, the first segment includes a
        header with the transcription model and detected language prepended
        to its "Text" field.

        When `return_language` is True, returns a tuple of
        `(segments, language_or_none)` for all providers. For Whisper,
        `language_or_none` is the detected language; for other providers it
        will typically be the `selected_source_lang` value or None.

        This function never returns plain text; callers that need a single
        transcript string should explicitly merge segment "Text" fields or use
        `transcribe_audio` instead.

        When persistence is enabled and a file path is provided, transcript
        cache files are stored under a dedicated `transcripts_cache` directory.
        If `base_dir` is provided, the cache is created under that directory;
        otherwise it is created under the system temp root.

    Raises:
        ValueError: If `audio_input` is not provided or is invalid.
        FileNotFoundError: If the audio path does not exist.
        RuntimeError: If transcription fails for other reasons (e.g., model loading
            error, issue during transcription process, or if no segments are produced).
            The original exception may be chained.
    """
    file_path: Optional[Path] = None
    file_path_label = "<memory>"
    time_start = time.time()
    base_dir_resolved: Optional[Path] = None

    if audio_input is None or (isinstance(audio_input, (str, Path)) and not str(audio_input)):
        log_counter("speech_to_text_error", labels={"error": "No audio input provided"})
        raise ValueError("speech-to-text: No audio input provided")

    # Parse the model name to determine the provider
    provider, model, variant = parse_transcription_model(whisper_model)
    hotwords_norm = _normalize_hotwords(hotwords)

    # Normalize task for Whisper-based providers. For non-Whisper providers,
    # this is ignored and we always perform transcription.
    task_normalized = str(task or "transcribe").strip().lower()
    if task_normalized not in {"transcribe", "translate"}:
        task_normalized = "transcribe"

    # If a file path is provided, resolve and validate it
    audio_path_for_model: Union[str, np.ndarray]
    if isinstance(audio_input, (str, Path)):
        raw_path = Path(audio_input)
        try:
            if base_dir is not None:
                base_dir_resolved = _resolve_allowed_base_dir(
                    base_dir,
                    label="Audio base directory",
                )
            candidate_path = raw_path
            if not candidate_path.is_absolute() and base_dir_resolved is not None:
                candidate_path = base_dir_resolved / candidate_path
            if candidate_path.exists():
                _assert_no_symlink(candidate_path, label="Audio input path")
            raw_path = _resolve_safe_input_path(
                raw_path,
                base_dir=base_dir_resolved,
                label="Audio input path",
            )
        except ValueError as exc:
            log_counter(
                "speech_to_text_error",
                labels={"error": "Audio path rejected", "file_path": str(raw_path)},
            )
            raise ValueError(str(exc)) from exc
        if not raw_path.exists():
            log_counter(
                "speech_to_text_error",
                labels={"error": "Audio file not found", "file_path": str(raw_path)},
            )
            raise FileNotFoundError(f"speech-to-text: Audio file not found at {raw_path}")
        _assert_no_symlink(raw_path, label="Audio input path")
        file_path = raw_path.resolve()
        file_path_label = str(file_path)
        audio_path_for_model = str(file_path)
    else:
        audio_np = np.asarray(audio_input, dtype=np.float32).flatten()
        sr = input_sample_rate or 16000
        audio_path_for_model = _resample_audio_if_needed(audio_np, sr, target_sr=16000)

    log_counter("speech_to_text_attempt", labels={"file_path": file_path_label, "model": whisper_model})
    _check_cancel(cancel_check, label="speech-to-text")

    # Route to the appropriate transcription provider (only supports file paths today)
    if provider == "parakeet":
        _check_cancel(cancel_check, label="speech-to-text")
        if file_path is None:
            raise ValueError("speech-to-text: Parakeet provider requires an audio file path")
        logging.info(f"Routing to Parakeet transcription with variant: {variant}")
        try:
            segments_parakeet = speech_to_text_parakeet(
                audio_file_path=str(file_path),
                variant=variant,
                selected_source_lang=selected_source_lang,
                vad_filter=vad_filter,
                base_dir=base_dir_resolved,
            )
            if return_language:
                return segments_parakeet, selected_source_lang
            return segments_parakeet
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
            normalized_variant = str(variant or "").strip().lower()
            if normalized_variant == "onnx":
                logging.error(f"Parakeet ONNX transcription failed (fail-fast): {e}")
                raise STTTranscriptionError(f"Parakeet ONNX transcription failed: {e}") from e
            logging.error(f"Parakeet transcription failed, falling back to whisper: {e}")
            provider = "whisper"
            model = "distil-large-v3"  # Default fallback model

    elif provider == "canary":
        _check_cancel(cancel_check, label="speech-to-text")
        if file_path is None:
            raise ValueError("speech-to-text: Canary provider requires an audio file path")
        logging.info("Routing to Canary transcription")
        try:
            segments_canary = speech_to_text_canary(
                audio_file_path=str(file_path),
                selected_source_lang=selected_source_lang,
                vad_filter=vad_filter,
                base_dir=base_dir_resolved,
            )
            if return_language:
                return segments_canary, selected_source_lang
            return segments_canary
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
            logging.error(f"Canary transcription failed, falling back to whisper: {e}")
            provider = "whisper"
            model = "distil-large-v3"

    elif provider == "qwen2audio":
        _check_cancel(cancel_check, label="speech-to-text")
        if file_path is None:
            raise ValueError("speech-to-text: Qwen2Audio provider requires an audio file path")
        logging.info("Routing to Qwen2Audio transcription")
        try:
            segments_qwen = speech_to_text_qwen2audio(
                audio_file_path=str(file_path),
                selected_source_lang=selected_source_lang,
                vad_filter=vad_filter,
                base_dir=base_dir_resolved,
            )
            if return_language:
                return segments_qwen, selected_source_lang
            return segments_qwen
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
            logging.error(f"Qwen2Audio transcription failed, falling back to whisper: {e}")
            provider = "whisper"
            model = "distil-large-v3"

    elif provider == "qwen3-asr":
        _check_cancel(cancel_check, label="speech-to-text")
        if file_path is None:
            raise ValueError("speech-to-text: Qwen3-ASR provider requires an audio file path")
        logging.info("Routing to Qwen3-ASR transcription")
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (  # type: ignore
                get_stt_provider_registry,
            )

            registry = get_stt_provider_registry()
            adapter = registry.get_adapter("qwen3-asr")
            model_name_for_provider = model or whisper_model
            artifact = adapter.transcribe_batch(
                str(file_path),
                model=model_name_for_provider,
                language=selected_source_lang,
                task=task_normalized,
                word_timestamps=word_timestamps,
                prompt=initial_prompt,
                hotwords=hotwords_norm,
                base_dir=base_dir_resolved,
                cancel_check=cancel_check,
            )
            segments_qwen3 = artifact.get("segments") or []
            language_qwen3 = artifact.get("language") or selected_source_lang
            if return_language:
                return segments_qwen3, language_qwen3
            return segments_qwen3
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
            logging.error(f"Qwen3-ASR transcription failed, falling back to whisper: {e}")
            provider = "whisper"
            model = "distil-large-v3"

    elif provider == "vibevoice":
        _check_cancel(cancel_check, label="speech-to-text")
        if file_path is None:
            raise ValueError("speech-to-text: VibeVoice-ASR provider requires an audio file path")
        logging.info("Routing to VibeVoice-ASR transcription")
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (  # type: ignore
                get_stt_provider_registry,
            )

            registry = get_stt_provider_registry()
            adapter = registry.get_adapter("vibevoice")
            model_name_for_provider = model or whisper_model
            artifact = adapter.transcribe_batch(
                str(file_path),
                model=model_name_for_provider,
                language=selected_source_lang,
                task=task_normalized,
                word_timestamps=word_timestamps,
                prompt=initial_prompt,
                hotwords=hotwords_norm,
                base_dir=base_dir_resolved,
                cancel_check=cancel_check,
            )
            segments_vibe = artifact.get("segments") or []
            language_vibe = artifact.get("language") or selected_source_lang
            if return_language:
                return segments_vibe, language_vibe
            return segments_vibe
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
            logging.error(f"VibeVoice-ASR transcription failed, falling back to whisper: {e}")
            provider = "whisper"
            model = "distil-large-v3"

    # If we get here, use the original whisper implementation
    # Update whisper_model to use the parsed model name (in case we fell back)
    if provider == "whisper":
        whisper_model = model

    logging.info(
        f"speech-to-text: Starting transcription for: {file_path_label}"
    )
    logging.info(f"speech-to-text: Model={whisper_model}, Lang={selected_source_lang or 'auto'}, VAD={vad_filter}")

    try:
        _check_cancel(cancel_check, label="speech-to-text")
        out_file = prettified_out_file = None
        cache_dir = None
        if file_path is not None:
            cache_dir = _resolve_transcript_cache_dir(
                file_path,
                base_dir=base_dir_resolved,
            )
            sanitized_whisper_model_name = sanitize_filename(whisper_model)
            out_file = cache_dir / f"{file_path.stem}-whisper_model-{sanitized_whisper_model_name}.segments.json"
            prettified_out_file = cache_dir / f"{file_path.stem}-whisper_model-{sanitized_whisper_model_name}.segments_pretty.json"

        options = {'beam_size': 5, 'best_of': 5, 'vad_filter': vad_filter}  # Simplified beam options
        if selected_source_lang:
            options["language"] = selected_source_lang
        if word_timestamps:
            options["word_timestamps"] = True

        # For Whisper, propagate the desired decoding task. Only "transcribe"
        # and "translate" are accepted; other values are coerced earlier.
        transcribe_options = dict(task=task_normalized, **options)
        combined_prompt = None
        if initial_prompt:
            combined_prompt = str(initial_prompt).strip() or None

        try:
            from .Audio_Custom_Vocabulary import initial_prompt_if_enabled
            _init_prompt = initial_prompt_if_enabled()
            if _init_prompt:
                _init_prompt = str(_init_prompt).strip()
                if _init_prompt:
                    combined_prompt = f"{_init_prompt}\n{combined_prompt}" if combined_prompt else _init_prompt
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as _cv_err:
            logging.debug(f"Custom vocab initial_prompt injection skipped: {_cv_err}")

        if combined_prompt:
            transcribe_options["initial_prompt"] = combined_prompt

        # Get model instance (cached)
        whisper_model_instance = get_whisper_model(
            whisper_model,
            processing_choice,
            check_download_status=False,
        )

        # Perform transcription (supports numpy arrays directly)
        _check_cancel(cancel_check, label="whisper transcribe")
        segments_raw, info = whisper_model_instance.transcribe(audio_path_for_model, **transcribe_options)

        detected_lang = getattr(info, "language", None)
        lang_prob = getattr(info, "language_probability", None)
        if detected_lang:
            logging.info(f"speech-to-text: Detected language: {detected_lang} (Confidence: {lang_prob:.2f})")

        segments = []
        for segment_chunk in segments_raw:
            _check_cancel(cancel_check, label="whisper segments")
            chunk = {
                "start_seconds": segment_chunk.start,
                "end_seconds": segment_chunk.end,
                "Time_Start": segment_chunk.start,
                "Time_End": segment_chunk.end,
                "Text": segment_chunk.text.strip() # Strip whitespace from text
            }
            if word_timestamps and hasattr(segment_chunk, 'words') and segment_chunk.words:
                try:
                    words_list = []
                    for w in segment_chunk.words:
                        words_list.append({
                            "start": float(w.start) if w.start is not None else None,
                            "end": float(w.end) if w.end is not None else None,
                            "word": getattr(w, 'word', getattr(w, 'text', '')).strip(),
                        })
                    if words_list:
                        chunk["words"] = words_list
                except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
                    pass
            logging.debug(f"Segment: {chunk}")
            try:
                from .Audio_Custom_Vocabulary import postprocess_text_if_enabled
                chunk["Text"] = postprocess_text_if_enabled(chunk["Text"]) or chunk["Text"]
            except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
                pass
            segments.append(chunk)
            logging.debug(f"Segment: [{chunk['start_seconds']:.2f}-{chunk['end_seconds']:.2f}] {chunk['Text'][:100]}...")

        if segments and include_metadata_header:
            segments[0]["Text"] = (
                f"This text was transcribed using whisper model: {whisper_model}\n"
                f"Detected language: {detected_lang}\n\n"
                f"{segments[0]['Text']}"
            )

        if not segments:
            log_counter("speech_to_text_error", labels={"error": "No transcription produced"})
            raise RuntimeError("No transcription produced. The audio may be invalid or empty.")

        transcription_time = time.time() - time_start
        logging.info(f"speech-to-text: Transcription completed in {transcription_time:.2f} seconds. Segments: {len(segments)}")
        log_histogram(
            "speech_to_text_duration",
            transcription_time,
            labels={"file_path": file_path_label, "model": whisper_model}
        )
        log_counter("speech_to_text_success", labels={"file_path": file_path_label, "model": whisper_model, "segments": len(segments)})

        # Persist transcription to disk for cache-aware callers (unless disabled)
        _persist_allowed = (file_path is not None) and (PERSIST_TRANSCRIPTS_DEFAULT if persist_segments is None else persist_segments)
        if _persist_allowed and out_file and prettified_out_file:
            try:
                _assert_no_symlink(out_file, label="Transcript cache file")
                payload = {"segments": segments}
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                try:
                    _assert_no_symlink(prettified_out_file, label="Transcript cache file")
                    with open(prettified_out_file, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as prettify_err:
                    logging.debug(f"Failed to write prettified transcription file: {prettify_err}")

                prune_transcript_cache(
                    cache_dir if cache_dir is not None else file_path.parent,
                    current_file=out_file,
                    max_age_days=cache_max_age_days if cache_max_age_days is not None else CACHE_MAX_AGE_DAYS,
                    max_total_mb=cache_max_total_mb if cache_max_total_mb is not None else CACHE_MAX_TOTAL_MB,
                    max_files_per_source=cache_max_files_per_source if cache_max_files_per_source is not None else CACHE_MAX_FILES_PER_SOURCE,
                )
            except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as persist_err:
                logging.warning(f"Could not persist transcription segments to cache: {persist_err}")

        gc.collect() # Suggest garbage collection
        if return_language:
            return segments, detected_lang
        return segments

    except TranscriptionCancelled:
        raise
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"speech-to-text: Error transcribing audio {file_path_label}: {e}", exc_info=True)
        log_counter(
            "speech_to_text_error",
            labels={"file_path": file_path_label, "model": whisper_model, "error": type(e).__name__}
        )
        if file_path is not None:
            raise RuntimeError(f"speech-to-text: Error during transcription of {file_path.name}") from e
        raise RuntimeError("speech-to-text: Error during transcription of in-memory audio") from e


def transcribe_audio_with_whisper(
    audio_file_path: str,
    diarize: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Backward-compatible helper expected by older workflow adapters/tests.

    Returns a dict with ``segments`` and ``duration`` using the same shape
    consumed by the diarization workflow fallback.
    """
    result = speech_to_text(audio_file_path, diarize=diarize, **kwargs)
    if isinstance(result, tuple):
        result = result[0]

    if isinstance(result, dict):
        return result

    if not isinstance(result, list):
        return {"segments": [], "duration": 0.0}

    segments: list[dict[str, Any]] = []
    max_end = 0.0
    for seg in result:
        if not isinstance(seg, dict):
            continue
        start = float(seg.get("start", seg.get("start_seconds", seg.get("Time_Start", 0.0))) or 0.0)
        end = float(seg.get("end", seg.get("end_seconds", seg.get("Time_End", start))) or start)
        text = str(seg.get("text") or seg.get("Text") or "")
        speaker = str(seg.get("speaker") or "SPEAKER_0")
        max_end = max(max_end, end)
        segments.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
            }
        )

    return {"segments": segments, "duration": max_end}

#
# End of Faster Whisper related functions
##########################################################

##########################################################
#
# Audio Conversion

class ConversionError(Exception):
    """Custom exception for errors during audio/video conversion."""
    pass


def _check_cancel(cancel_check: Optional[Callable[[], bool]], *, label: str) -> None:
    """Raise TranscriptionCancelled when cancel_check requests cancellation."""
    if cancel_check is None:
        return
    try:
        should_cancel = cancel_check()
        if inspect.isawaitable(should_cancel):
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = None
            if loop is not None and loop.is_running():
                with contextlib.suppress(_AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS):
                    should_cancel.close()
                raise CancelCheckError(
                    "_check_cancel received an awaitable cancel_check in the cancel_check handling branch "
                    "while an event loop is running; provide a synchronous cancel_check or update the API "
                    "to handle async cancel checks."
                )
            should_cancel = asyncio.run(should_cancel)
        if should_cancel:
            raise TranscriptionCancelled(f"Cancelled during {label}")
    except TranscriptionCancelled:
        raise
    except CancelCheckError:
        raise
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as exc:
        logging.error(f"cancel_check failed during {label}: {exc}", exc_info=True)
        raise CancelCheckError(f"cancel_check failed during {label}: {exc}") from exc

_FFMPEG_VERSION_CHECKED: bool = False
_FFMPEG_CMD_FOR_VERSION: Optional[str] = None


def _find_ffmpeg() -> str:
    """
    Finds the ffmpeg executable by checking common locations.

    Order of checks:
    1. Relative path: `../../Bin/ffmpeg.exe` (for Windows, specific to project structure).
    2. Environment variable: `FFMPEG_PATH`.
    3. System PATH: Uses `shutil.which("ffmpeg")`.

    Returns:
        The absolute path to the found ffmpeg executable as a string.

    Raises:
        FileNotFoundError: If ffmpeg is not found in any of the checked locations.
    """
    # 1. Check project Bin path (Windows) relative to repo structure
    if os.name == 'nt':
        try:
            app_dir = Path(__file__).resolve().parents[3]  # .../app
            bin_dir = app_dir / "Bin"
            ffmpeg_exe = bin_dir / "ffmpeg.exe"
            if ffmpeg_exe.exists():
                logging.debug(f"Found ffmpeg at project Bin path: {ffmpeg_exe}")
                return str(ffmpeg_exe)
        except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
            pass

    # 2. Check environment variable (useful for Docker/server setups)
    ffmpeg_env = os.environ.get("FFMPEG_PATH")
    if ffmpeg_env and Path(ffmpeg_env).exists():
        logging.debug(f"Found ffmpeg via FFMPEG_PATH env var: {ffmpeg_env}")
        return ffmpeg_env

    # 3. Project Bin path for non-Windows environments
    try:
        app_dir = Path(__file__).resolve().parents[3]
        candidate = app_dir / "Bin" / "ffmpeg"
        if candidate.exists():
            logging.debug(f"Found ffmpeg at project Bin path: {candidate}")
            return str(candidate)
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
        pass

    # 4. Check PATH using shutil.which (cross-platform)
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        logging.debug(f"Found ffmpeg in system PATH: {ffmpeg_path}")
        return ffmpeg_path

    # 5. If not found, raise error
    raise FileNotFoundError("ffmpeg executable not found in Bin directory, FFMPEG_PATH, or system PATH.")

# os.system(r'.\Bin\ffmpeg.exe -ss 00:00:00 -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{out_path}"')

def validate_audio_file(file_path: str, *, base_dir: Optional[Path] = None) -> tuple:
    """
    Validate audio file using ffprobe.

    Args:
        file_path: Path to the audio file to validate
        base_dir: Optional base directory used to validate local input paths.

    Returns:
        Tuple of (is_valid, error_message)

    Notes:
        This helper is a best-effort preflight check. When ffprobe is missing
        or misconfigured, unexpected exceptions are logged and the function
        returns ``(True, "Validation warning: ...")`` so that downstream
        conversion still runs and can surface more precise errors. The
        ``convert_to_wav`` helper applies stricter handling for clearly
        corrupted files (for example, zero channels or missing audio streams)
        and raises ``ConversionError`` in those cases.
    """
    try:
        # Check file exists and has minimum size
        try:
            path = _resolve_audio_input_path_for_provider(
                file_path,
                base_dir=base_dir,
                label="Audio input path",
            )
        except ValueError as exc:
            return False, str(exc)
        if not path.exists():
            return False, "File does not exist"

        stat = path.stat()
        file_size = stat.st_size
        if file_size < 1024:  # Less than 1KB
            return False, f"File too small ({file_size} bytes), likely corrupted"

        cache_key = str(path.resolve())
        cached = _AUDIO_VALIDATION_CACHE.get(cache_key)
        if cached and cached[0] == stat.st_mtime:
            return cached[1], cached[2]

        # Find ffprobe command
        ffmpeg_cmd = _find_ffmpeg()
        ffprobe_cmd = shutil.which("ffprobe")
        if not ffprobe_cmd:
            ffmpeg_path = Path(ffmpeg_cmd)
            suffix = ffmpeg_path.suffix  # '.exe' on Windows
            candidate = ffmpeg_path.with_name(f"ffprobe{suffix}")
            ffprobe_cmd = str(candidate) if candidate.exists() else "ffprobe"

        # Use ffprobe to check for audio streams
        result = subprocess.run(
            [ffprobe_cmd, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name,channels,sample_rate",
             "-of", "json", str(path)],
            capture_output=True,
            text=True,
            timeout=10  # 10 second timeout
        )

        if result.returncode != 0:
            return False, f"FFprobe failed: {result.stderr[:200] if result.stderr else 'Unknown error'}"

        # Parse JSON output
        try:
            import json
            probe_data = json.loads(result.stdout)
            if not probe_data.get('streams'):
                return False, "No audio streams detected in file"

            # Check first audio stream
            stream = probe_data['streams'][0]
            if stream.get('channels', 0) == 0:
                return False, "Audio stream has 0 channels"

        except (json.JSONDecodeError, KeyError):
            # If we can't parse the output, try simpler check
            if not result.stdout.strip():
                return False, "No audio stream detected"

        # Cache successful validation result keyed by file path + mtime so
        # repeated conversions of the same file avoid an extra ffprobe call.
        _AUDIO_VALIDATION_CACHE[cache_key] = (stat.st_mtime, True, "")
        return True, ""

    except subprocess.TimeoutExpired:
        return False, "File validation timed out (possible corrupt file)"
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
        logging.warning(f"Audio validation error: {e}")
        # Don't fail completely if validation has issues; allow FFmpeg-based
        # conversion/transcription to attempt processing and surface any errors.
        return True, f"Validation warning: {str(e)}"

#DEBUG
#@profile
@timeit
def convert_to_wav(
    video_file_path: str,
    offset: int = 0,
    end_time: Optional[int] = None,
    overwrite: bool = False,
    base_dir: Optional[Path] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> str:
    """
    Converts a video or audio file to a standardized WAV format using ffmpeg.

    The output WAV file is 16kHz, mono, 16-bit PCM signed little-endian,
    which is suitable for many speech recognition systems, including Whisper.
    The output file is saved in the same directory as the input file with
    a ".wav" extension.

    Args:
        video_file_path: The path to the input video or audio file.
        offset: The start offset in seconds from the beginning of the input
            file. ffmpeg's `-ss` parameter will be set to this value.
        end_time: Optional absolute end time in seconds. When provided, only the
            portion between `offset` and `end_time` is converted.
        overwrite: If True, overwrite the output WAV file if it already
            exists. If False and the file exists, the conversion is skipped,
            and the path to the existing file is returned.
        base_dir: Optional base directory used to validate local input/output
            paths. When provided, the input must resolve within this directory.
        cancel_check: Optional callable that returns True when conversion should
            be cancelled.

    Returns:
        The absolute path to the generated (or existing) WAV file as a string.

    Raises:
        FileNotFoundError: If the input `video_file_path` does not exist.
        RuntimeError: If the ffmpeg executable cannot be found or fails basic version check.
        ConversionError: If the ffmpeg conversion process fails (e.g., invalid input file,
            ffmpeg command returns non-zero exit code). This can also wrap other
            unexpected errors during ffmpeg execution.
    """
    log_counter("convert_to_wav_attempt", labels={"file_path": video_file_path})
    start_time = time.time()

    input_path = Path(video_file_path)
    try:
        input_path = _resolve_safe_input_path(
            input_path,
            base_dir=base_dir,
            label="Audio input path",
        )
    except ValueError as exc:
        raise ConversionError(str(exc)) from exc
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    try:
        _assert_no_symlink(input_path, label="Audio input path")
    except ValueError as exc:
        raise ConversionError(str(exc)) from exc
    _check_cancel(cancel_check, label="conversion preflight")

    # Output path in the same directory as input
    out_path = input_path.with_suffix(".wav")

    # Avoid in-place writes when input is already a .wav
    # If input and output resolve to the same file and overwrite is False, skip conversion.
    # If overwrite is True, choose a different output filename to prevent FFmpeg in-place editing.
    try:
        same_path = out_path.resolve() == input_path.resolve()
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS:
        same_path = str(out_path) == str(input_path)
    if same_path:
        if not overwrite:
            logging.info(
                f"Skipping conversion for WAV input (overwrite=False): {out_path}"
            )
            log_counter("convert_to_wav_skipped", labels={"file_path": video_file_path})
            return str(out_path)
        else:
            # Select a new output filename in the same directory
            alt_out_path = input_path.with_name(f"{input_path.stem}_16k_mono.wav")
            out_path = alt_out_path

    if out_path.exists():
        try:
            _assert_no_symlink(out_path, label="Audio output path")
        except ValueError as exc:
            raise ConversionError(str(exc)) from exc

    if out_path.exists() and not overwrite:
        logging.info(f"Skipping conversion as WAV file already exists and overwrite=False: {out_path}")
        log_counter("convert_to_wav_skipped", labels={"file_path": video_file_path})
        return str(out_path)

    if end_time is not None:
        if end_time <= offset:
            raise ConversionError(
                f"Invalid end time ({end_time}) provided; must be greater than offset ({offset})."
            )
        duration_seconds = end_time - offset
    else:
        duration_seconds = None

    # Determine ffmpeg executable path (honors FFMPEG_PATH and project Bin/)
    try:
        ffmpeg_cmd = _find_ffmpeg()
    except FileNotFoundError as e:
        error_msg = str(e)
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e

    # Verify ffmpeg command works, but avoid re-running the relatively
    # expensive `ffmpeg -version` probe on every conversion. We cache the
    # last-verified command and only re-check when the resolved executable
    # path changes (for example, if FFMPEG_PATH/env is updated).
    global _FFMPEG_VERSION_CHECKED, _FFMPEG_CMD_FOR_VERSION
    try:
        if not _FFMPEG_VERSION_CHECKED or ffmpeg_cmd != _FFMPEG_CMD_FOR_VERSION:
            subprocess.run(
                [ffmpeg_cmd, "-version"],
                check=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
            )
            logging.debug(f"Confirmed ffmpeg command '{ffmpeg_cmd}' is available.")
            _FFMPEG_VERSION_CHECKED = True
            _FFMPEG_CMD_FOR_VERSION = ffmpeg_cmd
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        error_msg = (
            f"ffmpeg command '{ffmpeg_cmd}' not found or failed execution. "
            "Please ensure ffmpeg is installed and in the system PATH or in the "
            f"expected ./Bin directory. Error: {e}"
        )
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e

    _check_cancel(cancel_check, label="conversion preflight")

    # Validate the audio file first (optional preflight). This can be disabled
    # via STT_SKIP_AUDIO_PREVALIDATION / [STT-Settings].skip_audio_prevalidation
    # for throughput-critical deployments that prefer to rely on ffmpeg alone.
    if not SKIP_AUDIO_PREVALIDATION:
        is_valid, validation_msg = validate_audio_file(str(input_path))
        if not is_valid:
            logging.warning(f"Audio file validation warning for '{input_path.name}': {validation_msg}")
            # For critical validation failures, we should fail early
            if "0 channels" in validation_msg or "No audio stream" in validation_msg or "too small" in validation_msg:
                logging.error(f"Critical audio file issue detected, cannot proceed with conversion: {validation_msg}")

                # Special handling for potential format mismatch (e.g., m4a file with .mp3 extension)
                if str(input_path).lower().endswith('.mp3') and "0 channels" in validation_msg:
                    logging.info("Possible format mismatch detected. Checking if file is actually a different format...")

                    # Try to detect actual format using ffprobe
                    try:
                        ffprobe_cmd = ffmpeg_cmd.replace('ffmpeg', 'ffprobe') if 'ffmpeg' in ffmpeg_cmd else 'ffprobe'
                        probe_result = subprocess.run(
                            [ffprobe_cmd, "-v", "error", "-show_entries", "format=format_name",
                             "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )

                        if probe_result.returncode == 0 and probe_result.stdout.strip():
                            detected_format = probe_result.stdout.strip()
                            logging.info(f"Detected actual format: {detected_format}")

                            # If it's actually m4a/mp4 audio, we can try to process it
                            if 'mp4' in detected_format or 'm4a' in detected_format or 'mov' in detected_format:
                                logging.info("File appears to be MP4/M4A format despite .mp3 extension. Will attempt conversion with format override.")
                                # Don't raise error, let FFmpeg try with proper format detection
                            else:
                                raise ConversionError(f"Audio file '{input_path.name}' is corrupted or invalid: {validation_msg}")
                        else:
                            raise ConversionError(f"Audio file '{input_path.name}' is corrupted or invalid: {validation_msg}")

                    except subprocess.TimeoutExpired:
                        logging.error("Format detection timed out")
                        raise ConversionError(f"Audio file '{input_path.name}' is corrupted or invalid: {validation_msg}") from None
                    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
                        logging.error(f"Error during format detection: {e}")
                        raise ConversionError(f"Audio file '{input_path.name}' is corrupted or invalid: {validation_msg}") from e
                else:
                    raise ConversionError(f"Audio file '{input_path.name}' is corrupted or invalid: {validation_msg}")
            # For other issues, continue anyway as FFmpeg might still handle it

    logging.info(f"Starting conversion to WAV: '{input_path.name}' -> '{out_path.name}'")

    # Enhanced command with better detection parameters
    # Don't force format based on extension - let FFmpeg auto-detect
    command = [
        ffmpeg_cmd,
        "-analyzeduration", "10M",    # Increase analysis duration for better format detection
        "-probesize", "50M",          # Increase probe size for difficult files
        "-i", str(input_path),        # Input file - FFmpeg will auto-detect format
        "-ss", str(offset),           # Use offset if needed (e.g., "00:00:10" or 10)
    ]
    if duration_seconds is not None:
        command.extend(["-t", str(duration_seconds)])
    command.extend([
        "-ar", "16000",               # Audio sample rate (good for Whisper)
        "-ac", "1",                   # Mono audio channel (good for Whisper)
        "-c:a", "pcm_s16le",          # Standard WAV audio codec
        "-y",                         # Overwrite output file without asking
        str(out_path)
    ])

    def _run_ffmpeg_command(cmd: list[str], *, label: str) -> subprocess.CompletedProcess:
        if cancel_check is None:
            return subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
            )
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            while True:
                try:
                    stdout, stderr = proc.communicate(timeout=0.5)
                    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
                except subprocess.TimeoutExpired:
                    try:
                        _check_cancel(cancel_check, label=label)
                    except TranscriptionCancelled:
                        try:
                            proc.terminate()
                            stdout, stderr = proc.communicate(timeout=2)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            stdout, stderr = proc.communicate()
                        raise
        finally:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as exc:
                    logging.debug(f"Failed to terminate ffmpeg process in {label}: {exc}")

    try:
        _check_cancel(cancel_check, label="ffmpeg conversion")
        result = _run_ffmpeg_command(command, label="ffmpeg conversion")

        # Check result
        if result.returncode != 0:
            # Log initial failure
            logging.warning(f"Initial conversion failed for '{input_path.name}', trying with more aggressive parameters")

            # Try with more lenient settings as fallback
            fallback_command = [
                ffmpeg_cmd,
                "-analyzeduration", "100M",   # Much higher analysis duration
                "-probesize", "100M",         # Much higher probe size
                "-err_detect", "ignore_err",  # Ignore errors and continue
                "-i", str(input_path),        # Let FFmpeg auto-detect format
                "-ss", str(offset),
            ]
            if duration_seconds is not None:
                fallback_command.extend(["-t", str(duration_seconds)])
            fallback_command.extend([
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                "-y",
                str(out_path)
            ])

            # Don't force format - let FFmpeg figure it out
            # The file extension might be wrong (e.g., m4a file with .mp3 extension)

            _check_cancel(cancel_check, label="ffmpeg fallback conversion")
            fallback_result = _run_ffmpeg_command(
                fallback_command,
                label="ffmpeg fallback conversion",
            )

            if fallback_result.returncode == 0:
                logging.info(f"Fallback conversion succeeded for '{input_path.name}'")
                result = fallback_result
                # Log any warnings from the fallback attempt
                if fallback_result.stderr:
                    logging.debug(f"Fallback conversion warnings: {fallback_result.stderr[:500]}")
            else:
                # Both attempts failed, use original error for reporting
                error_details = result.stderr or result.stdout or "No output captured"
                # Clean up potentially corrupted output file
                if out_path.exists():
                    with contextlib.suppress(OSError):
                        out_path.unlink()
                raise ConversionError(f"FFmpeg conversion failed (code {result.returncode}) for '{input_path.name}'. Error: {error_details.strip()}")
        else:
            logging.info(f"Conversion to WAV completed successfully: {out_path}")
            if result.stderr: # Log warnings even on success
                logging.warning(f"FFmpeg potential warnings for '{input_path.name}': {result.stderr.strip()}")
            log_counter("convert_to_wav_success", labels={"file_path": video_file_path})

    except TranscriptionCancelled:
         if out_path.exists():
             with contextlib.suppress(OSError):
                 out_path.unlink()
         raise
    except ConversionError:
         # Re-raise ConversionError explicitly to ensure it's caught
         log_counter("convert_to_wav_error", labels={"file_path": video_file_path, "error": "ffmpeg_failed"})
         raise
    except _AUDIO_TRANSCRIPTION_NONCRITICAL_EXCEPTIONS as e:
        # Catch other potential errors like permissions, etc.
        error_msg = f"Unexpected error during ffmpeg execution for '{input_path.name}': {e}"
        logging.error(error_msg, exc_info=True)
        log_counter("convert_to_wav_error", labels={"file_path": video_file_path, "error": str(e)})
        # Clean up potentially corrupted output file
        if out_path.exists():
             with contextlib.suppress(OSError):
                 out_path.unlink()
        raise ConversionError(error_msg) from e # Wrap other errors in ConversionError

    conversion_time = time.time() - start_time
    log_histogram("convert_to_wav_duration", conversion_time, labels={"file_path": video_file_path})

    gc.collect()
    return str(out_path)
#
# End of Audio Conversion Functions
##########################################################


##########################################################
#
# Transcript Handling/Processing

def format_transcription_with_timestamps(segments, keep_timestamps=True):
    """
    Formats the transcription segments with or without timestamps.
    Handles numeric seconds or pre-formatted HH:MM:SS strings.

    Parameters:
        segments (list): List of transcription segments (dicts with 'Time_Start', 'Time_End', 'Text').
        keep_timestamps (bool): Whether to include timestamps.

    Returns:
        str: Formatted transcription.
    """
    if not segments:
        return ""

    formatted_lines = []
    if keep_timestamps:
        for segment in segments:
            start = segment.get('Time_Start', 0)
            end = segment.get('Time_End', 0)
            text = segment.get('Text', '').strip()

            start_str = start
            end_str = end

            # Convert numeric seconds to HH:MM:SS if needed
            if isinstance(start, (int, float)):
                try:
                    start_str = time.strftime('%H:%M:%S', time.gmtime(float(start)))
                except (ValueError, TypeError, OSError): # Handle potential errors like large floats
                    start_str = f"{start:.2f}s" # Fallback to seconds
            if isinstance(end, (int, float)):
                try:
                    end_str = time.strftime('%H:%M:%S', time.gmtime(float(end)))
                except (ValueError, TypeError, OSError):
                    end_str = f"{end:.2f}s" # Fallback to seconds

            formatted_lines.append(f"[{start_str}-{end_str}] {text}")
    else:
        for segment in segments:
            text = segment.get('Text', '').strip()
            if text: # Avoid adding empty lines if a segment has no text
                formatted_lines.append(text)

    return "\n".join(formatted_lines)
#
# End of Transcript Handling/Processing
##########################################################


#
#
#######################################################################################################################
