# config.py
# Description: Configuration settings for the tldw server application.
#
from __future__ import annotations
# Imports
import configparser
import contextlib
import json
import os
from collections.abc import MutableMapping
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

#
# 3rd-party Libraries
from loguru import logger

from tldw_Server_API.app.core.config_paths import resolve_config_file
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_api_key_env_keys,
    custom_openai_config_option_names,
    custom_openai_endpoint_env_keys,
    custom_openai_model_env_keys,
    custom_openai_section_name,
    iter_custom_openai_provider_numbers,
)
from tldw_Server_API.app.core.testing import (
    env_flag_enabled,
    is_explicit_pytest_runtime,
    is_test_mode,
    is_truthy,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LlamaCppConfig
    from tldw_Server_API.app.core.config_sections import ConfigSections

_CONFIG_NONCRITICAL_EXCEPTIONS = (
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
    configparser.Error,
    yaml.YAMLError,
)


def _safe_json_dict(raw: Optional[str]) -> dict:
    if raw is None or str(raw).strip() == "":
        return {}
    try:
        parsed = json.loads(raw)
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        return {}
    return parsed if isinstance(parsed, dict) else {}


CUSTOM_OPENAI_ENDPOINT_ENV_KEYS = custom_openai_endpoint_env_keys(1)
CUSTOM_OPENAI2_ENDPOINT_ENV_KEYS = custom_openai_endpoint_env_keys(2)


def _first_nonempty_env(env_keys: tuple[str, ...]) -> Optional[str]:
    """Return the first non-empty environment value from ordered candidates."""
    for env_key in env_keys:
        value = os.getenv(env_key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _env_or_config_value(
    env_keys: tuple[str, ...],
    config_parser_object: configparser.ConfigParser,
    section: str,
    key: str,
    *,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """Resolve an env-first value with a single config option fallback."""
    return _first_nonempty_env(env_keys) or config_parser_object.get(section, key, fallback=fallback)


def _first_config_option_value(
    config_parser_object: configparser.ConfigParser,
    section: str,
    keys: tuple[str, ...],
    *,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """Return the first non-empty value from equivalent config option names."""
    sentinel = object()
    for key in keys:
        value = config_parser_object.get(section, key, fallback=sentinel)
        if value is sentinel or value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return fallback


def _env_or_config_option_value(
    env_keys: tuple[str, ...],
    config_parser_object: configparser.ConfigParser,
    section: str,
    keys: tuple[str, ...],
    *,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """Resolve ordered env aliases before ordered config option aliases."""
    return _first_nonempty_env(env_keys) or _first_config_option_value(
        config_parser_object,
        section,
        keys,
        fallback=fallback,
    )


def _int_env_or_cfg(
    env_value: Optional[object],
    cfg_value: Optional[object],
    default: int,
) -> int:
    def _clean(raw: Optional[object]) -> Optional[str]:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text or text.lower() in {"none", "null", "nil"}:
            return None
        return text

    env_text = _clean(env_value)
    if env_text is not None:
        return int(env_text)
    cfg_text = _clean(cfg_value)
    if cfg_text is not None:
        return int(cfg_text)
    return default

# Config.txt adapter cache + metadata
_CONFIG_PARSER_CACHE: Optional[configparser.ConfigParser] = None
_CONFIG_SOURCE_METADATA: dict[str, Any] = {
    "source": "config",
    "path": None,
    "loaded": False,
}

# Guard logging during module import so Loguru does not enqueue records before
# the import lock is released. Messages emitted before `_LOGGER_READY` flips to
# True are buffered and flushed once initialization completes.
_LOGGER_READY = False
_STARTUP_LOG_BUFFER: list[tuple[str, str, dict[str, Any]]] = []
_ENV_FILE_ENV_VAR = "TLDW_ENV_FILE"


def _buffered_log(level: str, message: str, **kwargs: Any) -> None:
    if _LOGGER_READY:
        logger.log(level, message, **kwargs)
    else:
        _STARTUP_LOG_BUFFER.append((level, message, kwargs))


def _log_info(message: str, **kwargs: Any) -> None:
    _buffered_log("INFO", message, **kwargs)


def _log_warning(message: str, **kwargs: Any) -> None:
    _buffered_log("WARNING", message, **kwargs)


def _log_error(message: str, **kwargs: Any) -> None:
    _buffered_log("ERROR", message, **kwargs)


def _log_critical(message: str, **kwargs: Any) -> None:
    _buffered_log("CRITICAL", message, **kwargs)


def _log_debug(message: str, **kwargs: Any) -> None:
    _buffered_log("DEBUG", message, **kwargs)


def _flush_startup_logs() -> None:
    global _STARTUP_LOG_BUFFER
    for level, message, kwargs in _STARTUP_LOG_BUFFER:
        logger.log(level, message, **kwargs)
    _STARTUP_LOG_BUFFER = []


def _resolve_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        return path.expanduser()


def get_tldw_env_file_path() -> Path | None:
    """Return the explicit runtime .env path selected by TLDW_ENV_FILE."""
    raw_path = os.getenv(_ENV_FILE_ENV_VAR)
    if not raw_path or not raw_path.strip():
        return None
    return _resolve_path(Path(raw_path.strip()))


def _candidate_env_paths(project_root: Path, repo_root: Path | None = None) -> list[Path]:
    """Return .env candidates in load precedence order.

    `TLDW_ENV_FILE` is an explicit user/runtime selection. It must be loaded
    before canonical repo paths because python-dotenv uses override=False.
    """
    candidates: list[Path] = []
    explicit_env_file = get_tldw_env_file_path()
    if explicit_env_file:
        candidates.append(explicit_env_file)

    candidates.extend(
        [
            project_root / 'Config_Files' / '.env',
            project_root / 'Config_Files' / '.ENV',
            project_root / '.env',
            project_root / '.ENV',
        ]
    )
    if repo_root is not None:
        candidates.extend(
            [
                repo_root / 'Config_Files' / '.env',
                repo_root / 'Config_Files' / '.ENV',
                repo_root / '.env',
                repo_root / '.ENV',
            ]
        )

    resolved_candidates: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = _resolve_path(path)
        key = str(resolved)
        if key in seen:
            continue
        resolved_candidates.append(resolved)
        seen.add(key)
    return resolved_candidates


def _load_env_files_early() -> None:
    """Load .env files before any environment reads.

    This mirrors the search locations used later in load_comprehensive_config(),
    but it is safe and no-op if the files do not exist. Keeping override=False
    ensures explicit environment variables are not replaced.
    """
    try:
        current_file_path = Path(__file__).resolve()
        project_root = current_file_path.parent.parent.parent
        candidate_env_paths = _candidate_env_paths(
            project_root,
            project_root.parent,
        )
        loaded_any = False
        for p in candidate_env_paths:
            try:
                if p.exists():
                    _log_info(f"Early loading environment variables from: {str(p)}")
                    load_dotenv(dotenv_path=str(p), override=False)
                    loaded_any = True
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                # Continue trying other candidates
                pass
        if not loaded_any:
            _log_debug("Early .env load: no candidate files found; relying on process env")
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        # Never fail early due to env file loading issues
        pass


# Load .env files immediately so module-level os.getenv(...) constants
# (including CORS allowlists) reflect Config_Files/.env values.
_load_env_files_early()


def _record_config_source(path: Optional[Path], loaded: bool) -> None:
    _CONFIG_SOURCE_METADATA.update(
        {
            "source": "config",
            "path": str(path) if path else None,
            "loaded": loaded,
        }
    )


def _load_config_parser(*, reload: bool = False) -> configparser.ConfigParser:
    global _CONFIG_PARSER_CACHE
    if _CONFIG_PARSER_CACHE is not None and not reload:
        return _CONFIG_PARSER_CACHE

    config_path_obj = resolve_config_file()
    config_parser = configparser.ConfigParser()

    if not config_path_obj.exists():
        _log_warning(
            f"Config file not found at {str(config_path_obj)}; using empty config"
        )
        _record_config_source(config_path_obj, loaded=False)
        _CONFIG_PARSER_CACHE = config_parser
        return config_parser

    try:
        config_parser.read(config_path_obj)
    except configparser.Error as e:
        _log_error(
            f"Error parsing config file {str(config_path_obj)}: {e}",
            exc_info=True,
        )
        raise

    _record_config_source(config_path_obj, loaded=True)
    _CONFIG_PARSER_CACHE = config_parser
    return config_parser


def get_config_section(section_name: str, *, reload: bool = False) -> dict[str, str]:
    """Return a config.txt section as a plain dict."""
    if reload:
        refresh_config_cache()
    parser = _load_config_parser()
    if not parser.has_section(section_name):
        return {}
    return dict(parser.items(section_name))


def get_config_value(
    section: str,
    key: str,
    default: Optional[str] = None,
    *,
    reload: bool = False,
) -> Optional[str]:
    """Return a config.txt value with a default fallback."""
    if reload:
        refresh_config_cache()
    parser = _load_config_parser()
    return parser.get(section, key, fallback=default)


ACTUAL_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_INGESTION_SOURCE_ALLOWED_ROOT_SECTION = "Files"
_INGESTION_SOURCE_ALLOWED_ROOT_KEY = "ingestion_source_allowed_roots"
_INGESTION_SOURCE_ALLOWED_ROOT_ENV_KEYS = (
    "INGESTION_SOURCE_ALLOWED_ROOTS",
    "TLDW_INGESTION_SOURCE_ALLOWED_ROOTS",
)


def _split_allowed_root_values(raw: Optional[object]) -> list[str]:
    if raw is None:
        return []
    parts: list[str] = []
    for chunk in str(raw).split(os.pathsep):
        for entry in chunk.split(","):
            value = entry.strip()
            if value:
                parts.append(value)
    return parts


def _normalize_allowed_root_entry(raw: object) -> Path | None:
    value = str(raw).strip()
    if not value:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (ACTUAL_PROJECT_ROOT / candidate).resolve(strict=False)
    else:
        candidate = candidate.resolve(strict=False)
    return candidate


def get_ingestion_source_allowed_roots(*, reload: bool = False) -> tuple[Path, ...]:
    roots: list[Path] = []
    seen: set[str] = set()

    config_value = get_config_value(
        _INGESTION_SOURCE_ALLOWED_ROOT_SECTION,
        _INGESTION_SOURCE_ALLOWED_ROOT_KEY,
        default="",
        reload=reload,
    )
    raw_values: list[str] = _split_allowed_root_values(config_value)
    for env_name in _INGESTION_SOURCE_ALLOWED_ROOT_ENV_KEYS:
        raw_values.extend(_split_allowed_root_values(os.getenv(env_name)))

    for raw_value in raw_values:
        normalized = _normalize_allowed_root_entry(raw_value)
        if normalized is None:
            continue
        marker = str(normalized)
        if marker in seen:
            continue
        seen.add(marker)
        roots.append(normalized)

    return tuple(roots)


def get_config_source_metadata() -> dict[str, Any]:
    """Return cached config source metadata for logging/diagnostics."""
    return dict(_CONFIG_SOURCE_METADATA)

# Local Imports
# Local Imports
#
########################################################################################################################
#
# Functions:

# --- Constants ---
# Client ID used by the Server API itself when writing to sync logs
SERVER_CLIENT_ID = "SERVER_API_V1"

# --- CORS Configuration ---
# List of allowed origins for CORS (env override supported)
def _parse_allowed_origins_env(raw: str):
    try:
        # Support JSON array
        if raw.strip().startswith("["):
            vals = json.loads(raw)
            return [str(v).strip() for v in vals if str(v).strip()]
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        pass
    # Fallback: comma-separated list
    return [s.strip() for s in raw.split(",") if s.strip()]

_DEFAULT_ALLOWED_ORIGINS = [
    # Common local origins
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://localhost:8081",
    "http://localhost:3000",  # Next.js frontend default
    "http://localhost:3001",  # Admin UI
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:8081",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://[::1]",
    "http://[::1]:8000",
    "http://[::1]:8080",
    "http://[::1]:8081",
    "http://[::1]:3000",
    "http://[::1]:3001",
    "https://localhost",
    "https://localhost:8080",
    "https://[::1]",
    "https://[::1]:8080",
]

_ENV_ALLOWED = os.getenv("ALLOWED_ORIGINS")
if _ENV_ALLOWED is None:
    ALLOWED_ORIGINS = list(_DEFAULT_ALLOWED_ORIGINS)
    _ALLOWED_ORIGINS_SOURCE = "default"
elif not str(_ENV_ALLOWED).strip():
    _log_warning(
        "ALLOWED_ORIGINS is set but empty; falling back to default local origins for compatibility."
    )
    ALLOWED_ORIGINS = list(_DEFAULT_ALLOWED_ORIGINS)
    _ALLOWED_ORIGINS_SOURCE = "default(empty-env)"
else:
    ALLOWED_ORIGINS = _parse_allowed_origins_env(_ENV_ALLOWED)
    _ALLOWED_ORIGINS_SOURCE = "env"


def get_default_allowed_origins() -> list[str]:
    """Return the built-in local/loopback CORS origins used for self-hosted browser access."""
    return list(_DEFAULT_ALLOWED_ORIGINS)

# --- API Configuration ---
# API version prefix for all endpoints
API_V1_PREFIX = "/api/v1"

# Authentication prefix
AUTH_BEARER_PREFIX = "Bearer "

# --- Server Configuration ---
# Default server host and port
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# --- Database Configuration ---
# Default database type
DEFAULT_DB_TYPE = "sqlite3"

# --- File Validation/YARA Settings ---
YARA_RULES_PATH: Optional[str] = None # e.g., "/app/yara_rules/index.yar"
MAGIC_FILE_PATH: Optional[str] = os.getenv("MAGIC_FILE_PATH", None) # e.g., "/app/magic.mgc"

# --- Chunking Settings ---
global_default_chunk_language = "en"


# TTS-related legacy defaults (placeholder-free; real secrets come from env/config files)
APP_CONFIG = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "KOKORO_ONNX_MODEL_PATH_DEFAULT": "models/kokoro/onnx/model.onnx",
    "KOKORO_ONNX_VOICES_JSON_DEFAULT": "models/kokoro/voices",
    "KOKORO_DEVICE_DEFAULT": "cpu", # or "cuda"
    "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY", ""),
    "local_kokoro_default_onnx": { # Specific overrides for this backend_id
        "KOKORO_DEVICE": "cuda:0"
    },
    "global_tts_settings": {
        # shared settings
    }
}

DATABASE_CONFIG = {
    }

RAG_SEARCH_CONFIG = {
    "fts_top_k": 10,
    "vector_top_k": 10,
    "web_vector_top_k": 10,
    "llm_context_document_limit": 10,
    "chat_context_limit": 10,
    # Database query limits
    "max_conversations_per_character": 1000,
    "max_conversations_for_keyword": 500,
    "max_notes_for_keyword": 500,
    "max_character_cards_fetch": 100000,
    "max_notes_fetch": 100000,
    "max_media_search_limit": 10000,
    # Embedding and vector search
    "max_embedding_batch_size": 100,
    "max_vector_search_results": 1000,
    # Content limits
    "max_context_chars_rag": 15000,
    "metadata_content_preview_chars": 256,
    # Cleanup settings
    "temp_file_cleanup_hours": 24,
    # Pagination defaults
    "default_results_per_page": 50,
}

# Configuration for the new modular RAG service
RAG_SERVICE_CONFIG = {
    # General settings
    "batch_size": 32,
    "num_workers": 4,  # Limited for multi-user server
    "log_level": "INFO",
    "log_performance_metrics": True,

    # Cache configuration
    "cache": {
        "enable_cache": True,
        "max_cache_size": 1000,
        "cache_ttl": 3600,  # 1 hour
        "cache_search_results": True,
        "cache_embeddings": True,
        "cache_llm_responses": False  # Don't cache LLM responses by default
    },

    # Retriever configuration
    "retriever": {
        "fts_top_k": 10,
        "vector_top_k": 10,
        "hybrid_alpha": 0.5,  # Balance between keyword and vector search
        "enable_web_search": False,
        "web_search_top_k": 5,
        "enable_re_ranking": True,
        "re_ranking_model": "flashrank",
        "timeout_seconds": 30
    },

    # Processor configuration
    "processor": {
        "enable_reranking": True,
        "reranking_model": "flashrank",
        "reranking_top_k": 10,
        "enable_deduplication": True,
        "dedup_threshold": 0.85,
        "max_context_length": 4096,
        "context_padding_tokens": 100,
        "enable_metadata_filtering": True,
        # Tokenizer name, not a secret.
        "token_counter": "tiktoken"  # nosec B105
    },

    # Generator configuration
    "generator": {
        "default_model": "gpt-3.5-turbo",
        "default_temperature": 0.7,
        "default_max_tokens": 2000,
        "enable_streaming": True,
        "stream_chunk_size": 50,
        "enable_citations": True,
        "citation_style": "inline",
        "enable_fallback": True,
        "fallback_behavior": "simple_answer"
    }
}

# Configuration for speaker diarization (config-level defaults)
DIARIZATION_CONFIG = {
    # Backend selection
    "backend": "embedding",  # embedding | nemo_multitalk

    # VAD (Voice Activity Detection) settings
    "vad_backend": "silero_hub",  # silero_hub | onnx_silero
    "vad_threshold": 0.5,  # Silero VAD confidence threshold
    "vad_min_speech_duration": 0.25,  # seconds
    "vad_min_silence_duration": 0.25,  # seconds
    "allow_vad_fallback": True,
    "enable_torch_hub_fetch": True,
    "speech_pad_ms": 400,  # Padding around speech segments in milliseconds
    "onnx_model_path": "models/silero_vad/silero_vad_v6.onnx",

    # Segmentation settings
    "segment_duration": 30.0,  # Maximum segment duration in seconds
    "segment_overlap": 0.5,
    "min_segment_duration": 1.0,
    "max_segment_duration": 3.0,

    # Embedding model settings
    "embedding_model": "speechbrain/spkrec-ecapa-voxceleb",
    "embedding_batch_size": 32,
    "embedding_device": "auto",  # auto, cpu, cuda, or cuda:0
    "embedding_local_only": False,

    # Clustering settings
    "clustering_method": "spectral",  # spectral or agglomerative
    "similarity_threshold": 0.85,
    "min_speakers": 1,
    "max_speakers": 10,

    # Post-processing settings
    "merge_threshold": 0.5,  # Threshold for merging adjacent segments
    "min_speaker_duration": 3.0,  # Minimum total speaker duration in seconds
    "detect_overlapping_speech": False,
    "overlap_confidence_threshold": 0.7,

    # Performance / memory settings
    "num_threads": 4,  # Number of threads for processing
    "memory_efficient": False,
    "max_memory_mb": 2048,
    # Default sentinel; real HuggingFace auth token is supplied via config/env.
    "use_auth_token": None,  # nosec B105
    "cache_dir": None,  # Directory for model cache

    # Output settings
    "include_embeddings": False,  # Include embeddings in output
    "include_vad_scores": False,  # Include VAD scores in output

    # NeMo multitalk (Parakeet + Sortformer diarization)
    "nemo_multitalk_asr_model": "nvidia/multitalker-parakeet-streaming-0.6b-v1",
    "nemo_multitalk_diar_model": "nvidia/diar_streaming_sortformer_4spk-v2.1",
    "nemo_multitalk_device": "auto",  # auto, cpu, cuda
    "nemo_multitalk_cache_dir": None,
    "nemo_multitalk_max_speakers": 4,
    "nemo_multitalk_disable_cuda_graphs": True,
}


def load_tts_config() -> dict[str, Any]:
    """
    Load TTS configuration from YAML file and integrate with existing config.

    Returns:
        Dictionary containing TTS configuration
    """
    current_file_path = Path(__file__).resolve()
    candidate_paths = [
        current_file_path.parent / 'TTS' / 'tts_providers_config.yaml',
        current_file_path.parent.parent / 'Config_Files' / 'tts_providers_config.yaml',
        Path.cwd() / 'tldw_Server_API' / 'Config_Files' / 'tts_providers_config.yaml',
    ]
    tts_config_path = None
    for p in candidate_paths:
        if p.exists():
            tts_config_path = p
            break
    if tts_config_path is None:
        tts_config_path = candidate_paths[0]

    _log_info(f"Loading TTS configuration from: {tts_config_path}")

    if not tts_config_path.exists():
        _log_warning(f"TTS config file not found at {tts_config_path}, using defaults")
        return _get_default_tts_config()

    try:
        with open(tts_config_path, encoding='utf-8') as f:
            tts_config = yaml.safe_load(f)

        # Validate and process the configuration
        processed_config = _process_tts_config(tts_config)
        _log_info("TTS configuration loaded successfully")
        return processed_config

    except _CONFIG_NONCRITICAL_EXCEPTIONS as e:
        _log_error(f"Error loading TTS configuration: {e}")
        _log_info("Falling back to default TTS configuration")
        return _get_default_tts_config()

def _process_tts_config(tts_config: dict[str, Any]) -> dict[str, Any]:
    """
    Process and validate TTS configuration from YAML.

    Args:
        tts_config: Raw configuration from YAML file

    Returns:
        Processed configuration dictionary
    """
    processed = {}

    # Extract provider priority
    if 'provider_priority' in tts_config:
        processed['provider_priority'] = tts_config['provider_priority']

    # Process provider configurations
    if 'providers' in tts_config:
        for provider_name, provider_config in tts_config['providers'].items():
            processed[f'{provider_name}_config'] = provider_config

            # Enable/disable flags
            processed[f'{provider_name}_enabled'] = provider_config.get('enabled', True)

    # Extract voice mappings
    if 'voice_mappings' in tts_config:
        processed['voice_mappings'] = tts_config['voice_mappings']

    # Extract format preferences
    if 'format_preferences' in tts_config:
        processed['format_preferences'] = tts_config['format_preferences']

    # Extract performance settings
    if 'performance' in tts_config:
        processed.update(tts_config['performance'])

    # Extract fallback settings
    if 'fallback' in tts_config:
        processed.update(tts_config['fallback'])

    # Extract logging settings
    if 'logging' in tts_config:
        processed['tts_logging'] = tts_config['logging']

    return processed

def _get_default_tts_config() -> dict[str, Any]:
    """
    Get default TTS configuration when YAML config is not available.

    Returns:
        Default configuration dictionary
    """
    return {
        'provider_priority': ['openai', 'kokoro'],
        'openai_enabled': True,
        'kokoro_enabled': True,
        'higgs_enabled': False,
        'dia_enabled': False,
        'chatterbox_enabled': False,
        'max_concurrent_generations': 4,
        'fallback_enabled': True,
        'max_attempts': 3,
        'retry_delay_ms': 1000
    }

def load_openai_mappings() -> dict:
    # Determine path relative to this file.
    # config.py is in project_root/tldw_server_api/app/core/config.py
    # Config_Files is assumed to be in project_root/tldw_server_api/Config_Files/
    current_file_path = Path(__file__).resolve()
    api_component_root = current_file_path.parent.parent.parent  # This should be /project_root/tldw_server_api/

    mapping_path = api_component_root / "Config_Files" / "openai_tts_mappings.json"
    _log_debug(f"Attempting to load OpenAI TTS mappings from: {str(mapping_path)}")
    try:
        with open(mapping_path) as f:
            return json.load(f)
    except _CONFIG_NONCRITICAL_EXCEPTIONS as e:
        _log_debug(f"Failed to load OpenAI TTS mappings from {mapping_path}: {e}")
        # Fallback to a default or raise an error
        return {
            "models": {"tts-1": "openai_official_tts-1"},
            "voices": {"alloy": "alloy"}
        }

_openai_mappings = load_openai_mappings()

openai_tts_mappings = {
    "models": {
        "tts-1": "openai_official_tts-1",
        "tts-1-hd": "openai_official_tts-1-hd",
        "eleven_monolingual_v1": "elevenlabs_english_v1",
        "kokoro": "local_kokoro_default_onnx"
    },
    "voices": {
        "alloy": "alloy", "echo": "echo", "fable": "fable",
        "onyx": "onyx", "nova": "nova", "shimmer": "shimmer",

        "RachelEL": "21m00Tcm4TlvDq8ikWAM",

        "k_bella": "af_bella",
        "k_adam" : "am_v0adam"
    }
}


# --- Helper Function (Optional but can keep dictionary creation clean) ---
def load_settings():
    """
    Assembles application configuration from environment variables and configuration files into a single mapping.

    Builds a consolidated dictionary of runtime settings (paths, feature flags, numeric limits, provider blocks, and raw comprehensive config) used by the application. The returned mapping contains keys such as PROJECT_ROOT, DATABASE_URL, USER_DB_BASE_DIR, REDIS_*, SANDBOX_*, feature-flag knobs, and COMPREHENSIVE_CONFIG_RAW among many others.

    Returns:
        dict: A dictionary of consolidated configuration values keyed by setting name.

    Notes:
        This function may load .env files, consult on-disk config files, emit startup warnings, and create filesystem directories (for the main SQLite database and the user data base directory) as needed.
    """

    _log_info(f"Determined ACTUAL_PROJECT_ROOT for database paths: {ACTUAL_PROJECT_ROOT}")

    # Ensure .env files are loaded before reading any environment variables
    _load_env_files_early()

    # --- Application Mode ---
    single_user_mode_str = os.getenv("APP_MODE", "single").lower()
    single_user_mode = single_user_mode_str != "multi"

    # --- Single-User Settings ---
    # Use a fixed ID for the single user's database path and cache key
    # Default to 1 for single-user mode (0 is typically reserved for system/admin)
    single_user_fixed_id = int(os.getenv("SINGLE_USER_FIXED_ID", "1")) # Default to user ID 1
    # API Key for accessing the single-user instance
    # Check both SINGLE_USER_API_KEY (AuthNZ standard) and API_KEY (legacy) environment variables
    single_user_api_key = os.getenv("SINGLE_USER_API_KEY") or os.getenv("API_KEY")

    # --- Multi-User Settings (JWT) ---
    jwt_secret_key = os.getenv("JWT_SECRET_KEY", "a_very_insecure_default_secret_key_for_dev_only")
    jwt_algorithm = "HS256"
    access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # --- Redis Configuration ---
    # Initialize comprehensive_config early to avoid UnboundLocalError
    try:
        comprehensive_config: dict[str, Any] = load_and_log_configs() or {}
    except _CONFIG_NONCRITICAL_EXCEPTIONS as e:
        _log_error(f"Error loading comprehensive_config: {e}", exc_info=True)
        comprehensive_config = {}
    def _redis_section_get(key: str, default: Optional[str] = None) -> Optional[str]:
        section = comprehensive_config.get('Redis') or {}
        try:
            if hasattr(section, "get"):
                return section.get(key, fallback=default)  # type: ignore[arg-type]
        except TypeError:
            # Some objects (e.g., dict) don't support fallback kwarg
            pass
        if isinstance(section, dict):
            return section.get(key, default)
        return default
    def _to_bool(value: Optional[str], default: bool = False) -> bool:
        if value is None:
            return default
        return is_truthy(value)

    def _safe_int(value: object, default: int) -> int:
        try:
            if value is None:
                return default
            s = str(value).strip()
            if not s:
                return default
            return int(s)
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            return default

    def _env_or_cfg_bool(env_key: str, cfg_key: str, default: bool) -> bool:
        env_val = os.getenv(env_key)
        if env_val is not None:
            return is_truthy(env_val)
        cfg_val = get_config_value("TTS-Settings", cfg_key)
        if cfg_val is None:
            return default
        return is_truthy(cfg_val)

    def _env_or_cfg_int(env_key: str, cfg_key: str, default: int) -> int:
        env_val = os.getenv(env_key)
        if env_val is not None:
            return _safe_int(env_val, default)
        cfg_val = get_config_value("TTS-Settings", cfg_key)
        if cfg_val is None:
            return default
        return _safe_int(cfg_val, default)

    def _env_or_cfg_str(env_key: str, cfg_key: str, default: Optional[str]) -> Optional[str]:
        env_val = os.getenv(env_key)
        if env_val is not None:
            s = str(env_val).strip()
            return s if s else default
        cfg_val = get_config_value("TTS-Settings", cfg_key)
        if cfg_val is None:
            return default
        s = str(cfg_val).strip()
        return s if s else default

    # Load from comprehensive config first, allowing env overrides
    redis_host = os.getenv("REDIS_HOST") or _redis_section_get('redis_host', 'localhost') or 'localhost'
    redis_port_raw = os.getenv("REDIS_PORT") or _redis_section_get('redis_port', '6379') or '6379'
    redis_db_raw = os.getenv("REDIS_DB") or _redis_section_get('redis_db', '0') or '0'
    cache_ttl_raw = os.getenv("CACHE_TTL") or _redis_section_get('cache_ttl', '300') or '300'
    redis_enabled_env = os.getenv("REDIS_ENABLED")
    # Enable Redis by default on localhost so integration tests attempt connection
    redis_enabled = _to_bool(redis_enabled_env, _to_bool(_redis_section_get('redis_enabled', 'true'), True))

    try:
        redis_port = int(str(redis_port_raw))
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        redis_port = 6379
    try:
        redis_db = int(str(redis_db_raw))
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        redis_db = 0
    try:
        cache_ttl = int(str(cache_ttl_raw))
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        cache_ttl = 300

    config_redis_url = _redis_section_get('redis_url')
    redis_url_env = os.getenv("REDIS_URL")

    if redis_url_env:
        redis_url = redis_url_env
    elif config_redis_url:
        redis_url = config_redis_url
        try:
            parsed = urlparse(redis_url)
            if parsed.hostname and not os.getenv("REDIS_HOST"):
                redis_host = parsed.hostname
            if parsed.port and not os.getenv("REDIS_PORT"):
                redis_port = parsed.port
            if parsed.path and parsed.path != "/" and not os.getenv("REDIS_DB"):
                with contextlib.suppress(_CONFIG_NONCRITICAL_EXCEPTIONS):
                    redis_db = int(parsed.path.lstrip("/"))
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            pass
    else:
        redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"

    tts_history_enabled = _env_or_cfg_bool("TTS_HISTORY_ENABLED", "tts_history_enabled", True)
    tts_history_store_text = _env_or_cfg_bool("TTS_HISTORY_STORE_TEXT", "tts_history_store_text", True)
    tts_history_store_failed = _env_or_cfg_bool("TTS_HISTORY_STORE_FAILED", "tts_history_store_failed", True)
    tts_history_hash_key = _env_or_cfg_str("TTS_HISTORY_HASH_KEY", "tts_history_hash_key", None)
    tts_history_retention_days = _env_or_cfg_int("TTS_HISTORY_RETENTION_DAYS", "tts_history_retention_days", 90)
    tts_history_max_rows_per_user = _env_or_cfg_int("TTS_HISTORY_MAX_ROWS_PER_USER", "tts_history_max_rows_per_user", 10000)
    tts_history_purge_interval_hours = _env_or_cfg_int(
        "TTS_HISTORY_PURGE_INTERVAL_HOURS",
        "tts_history_purge_interval_hours",
        24,
    )

    # Base directory for all user-specific data (USER_DB_BASE_DIR default): ACTUAL_PROJECT_ROOT/Databases/user_databases/
    default_user_data_base_dir = ACTUAL_PROJECT_ROOT / "Databases" / "user_databases"
    config_user_db_base_dir = None
    try:
        config_user_db_base_dir = get_config_value("TTS-Settings", "USER_DB_BASE_DIR")
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        config_user_db_base_dir = None
    if config_user_db_base_dir is not None:
        config_user_db_base_dir = str(config_user_db_base_dir).strip() or None
    env_user_db_base_dir = os.getenv("USER_DB_BASE_DIR")
    user_data_base_dir_str = (
        config_user_db_base_dir
        or env_user_db_base_dir
        or str(default_user_data_base_dir.resolve())
    )
    user_data_base_dir = Path(user_data_base_dir_str)

    # Main/central SQLite database: ACTUAL_PROJECT_ROOT/Databases/user_databases/<SINGLE_USER_FIXED_ID>/tldw.db
    default_main_db_path = (ACTUAL_PROJECT_ROOT / "Databases" / "user_databases" / f"{single_user_fixed_id}" / "tldw.db").resolve()
    default_database_url = f"sqlite:///{default_main_db_path}"
    database_url = os.getenv("DATABASE_URL", default_database_url)

    users_db_configured = os.getenv("USERS_DB_ENABLED", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Audit settings (env overrides config.txt [Audit])
    audit_stream_env_raw = os.getenv("AUDIT_EXPORT_STREAM_AUTO_MAX_ROWS")
    audit_storage_env_raw = os.getenv("AUDIT_STORAGE_MODE")
    audit_shared_env_raw = os.getenv("AUDIT_SHARED_DB_PATH")
    audit_rollback_env_raw = os.getenv("AUDIT_STORAGE_ROLLBACK")
    audit_etl_subpath_env_raw = os.getenv("AUDIT_ETL_USER_SUBPATH")

    audit_stream_cfg_raw: Optional[str] = None
    audit_storage_cfg_raw: Optional[str] = None
    audit_shared_cfg_raw: Optional[str] = None
    audit_rollback_cfg_raw: Optional[str] = None
    audit_etl_subpath_cfg_raw: Optional[str] = None
    _audit_parser = None
    if (
        audit_stream_env_raw is None
        or audit_storage_env_raw is None
        or audit_shared_env_raw is None
        or audit_rollback_env_raw is None
        or audit_etl_subpath_env_raw is None
    ):
        try:
            _audit_parser = load_comprehensive_config()
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            _audit_parser = None
    if _audit_parser is not None:
        try:
            if hasattr(_audit_parser, "has_section") and _audit_parser.has_section("Audit"):
                if audit_stream_env_raw is None:
                    audit_stream_cfg_raw = _audit_parser.get("Audit", "export_stream_auto_max_rows", fallback=None)
                if audit_storage_env_raw is None:
                    audit_storage_cfg_raw = _audit_parser.get("Audit", "storage_mode", fallback=None)
                if audit_shared_env_raw is None:
                    audit_shared_cfg_raw = _audit_parser.get("Audit", "shared_db_path", fallback=None)
                if audit_rollback_env_raw is None:
                    audit_rollback_cfg_raw = _audit_parser.get("Audit", "storage_rollback", fallback=None)
                if audit_etl_subpath_env_raw is None:
                    audit_etl_subpath_cfg_raw = _audit_parser.get("Audit", "etl_user_subpath", fallback=None)
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            audit_stream_cfg_raw = None
            audit_storage_cfg_raw = None
            audit_shared_cfg_raw = None
            audit_rollback_cfg_raw = None
            audit_etl_subpath_cfg_raw = None

    audit_stream_auto_max_rows = _safe_int(
        audit_stream_env_raw if audit_stream_env_raw is not None else audit_stream_cfg_raw,
        5000,
    )

    audit_storage_mode_raw = audit_storage_env_raw if audit_storage_env_raw is not None else audit_storage_cfg_raw
    audit_storage_mode = str(audit_storage_mode_raw).strip().lower() if audit_storage_mode_raw else "per_user"
    if audit_storage_mode not in {"per_user", "shared"}:
        audit_storage_mode = "per_user"

    audit_shared_db_path_raw = audit_shared_env_raw if audit_shared_env_raw is not None else audit_shared_cfg_raw
    audit_shared_db_path = (
        str(audit_shared_db_path_raw).strip()
        if audit_shared_db_path_raw is not None and str(audit_shared_db_path_raw).strip()
        else "Databases/audit_shared.db"
    )

    audit_storage_rollback_raw = (
        audit_rollback_env_raw if audit_rollback_env_raw is not None else audit_rollback_cfg_raw
    )
    audit_storage_rollback = _to_bool(
        str(audit_storage_rollback_raw) if audit_storage_rollback_raw is not None else None,
        False,
    )
    audit_etl_subpath_raw = (
        audit_etl_subpath_env_raw if audit_etl_subpath_env_raw is not None else audit_etl_subpath_cfg_raw
    )
    audit_etl_user_subpath = (
        str(audit_etl_subpath_raw).strip()
        if audit_etl_subpath_raw is not None and str(audit_etl_subpath_raw).strip()
        else None
    )

    # Load comprehensive configurations (API keys, embedding settings, etc.)
    # If SINGLE_USER_API_KEY wasn't present before, try reading again now that configs/.env may be loaded
    if not single_user_api_key:
        with contextlib.suppress(_CONFIG_NONCRITICAL_EXCEPTIONS):
            single_user_api_key = os.getenv("SINGLE_USER_API_KEY") or os.getenv("API_KEY")


    content_backend_mode = os.getenv("TLDW_CONTENT_DB_BACKEND", "sqlite").strip().lower()

    # Character-Chat rate limiting knobs from config.txt (env will still override later)
    try:
        _cp = load_comprehensive_config()
        def _cc_int(key: str, fb: int) -> int:
            try:
                return int(str(_cp.get('Character-Chat', key, fallback=str(fb))))
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                return fb
        def _cc_bool_present(key: str) -> tuple[bool, bool]:
            try:
                if _cp.has_section('Character-Chat') and _cp.has_option('Character-Chat', key):
                    raw = str(_cp.get('Character-Chat', key)).strip().lower()
                    return True, is_truthy(raw)
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                pass
            return False, False
        _character_rate_limit_ops = _cc_int('CHARACTER_RATE_LIMIT_OPS', 100)
        _character_rate_limit_window = _cc_int('CHARACTER_RATE_LIMIT_WINDOW', 3600)
        _max_characters_per_user = _cc_int('MAX_CHARACTERS_PER_USER', 1000)
        _max_character_import_size_mb = _cc_int('MAX_CHARACTER_IMPORT_SIZE_MB', 10)
        # Optional chat-specific knobs if present
        _max_chats_per_user = _cc_int('MAX_CHATS_PER_USER', 100000)
        _max_messages_per_chat = _cc_int('MAX_MESSAGES_PER_CHAT', 1000)
        _max_messages_per_chat_soft = _cc_int('MAX_MESSAGES_PER_CHAT_SOFT', _max_messages_per_chat)
        _max_chat_completions_per_minute = _cc_int('MAX_CHAT_COMPLETIONS_PER_MINUTE', 20)
        _max_message_sends_per_minute = _cc_int('MAX_MESSAGE_SENDS_PER_MINUTE', 60)
        _has_char_rl_enabled, _char_rl_enabled_bool = _cc_bool_present('CHARACTER_RATE_LIMIT_ENABLED')
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        _character_rate_limit_ops = 100
        _character_rate_limit_window = 3600
        _max_characters_per_user = 10000
        _max_character_import_size_mb = 10
        _max_chats_per_user = 100000
        _max_messages_per_chat = 1000
        _max_messages_per_chat_soft = _max_messages_per_chat
        _max_chat_completions_per_minute = 20
        _max_message_sends_per_minute = 60
        _has_char_rl_enabled = False
        _char_rl_enabled_bool = False

    # -------------------------
    # Sandbox (Code Interpreter) Settings
    # -------------------------
    try:
        cp = load_comprehensive_config()
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        cp = None

    def _sbx_get(key: str, fallback: Optional[str] = None) -> Optional[str]:
        try:
            if cp and hasattr(cp, "has_section") and cp.has_section('Sandbox'):
                # type: ignore[no-untyped-call]
                return cp.get('Sandbox', key, fallback=fallback)  # type: ignore[arg-type]
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            pass
        return fallback

    def _sbx_env_or_cfg(env_key: str, cfg_key: str, default: str) -> str:
        return os.getenv(env_key) or _sbx_get(cfg_key, default) or default

    def _sbx_int(env_key: str, cfg_key: str, default: int) -> int:
        raw = os.getenv(env_key) or _sbx_get(cfg_key, str(default)) or str(default)
        try:
            return int(str(raw))
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            return default

    def _sbx_float(env_key: str, cfg_key: str, default: float) -> float:
        raw = os.getenv(env_key) or _sbx_get(cfg_key, str(default)) or str(default)
        try:
            return float(str(raw))
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            return default

    def _sbx_list(env_key: str, cfg_key: str, default: list[str]) -> list[str]:
        raw_env = os.getenv(env_key)
        if raw_env is not None:
            try:
                if raw_env.strip().startswith("["):
                    import json as _json
                    vals = _json.loads(raw_env)
                    return [str(v).strip() for v in vals if str(v).strip()]
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                pass
            return [s.strip() for s in raw_env.split(',') if s.strip()]
        raw_cfg = _sbx_get(cfg_key, None)
        if raw_cfg is not None:
            try:
                if raw_cfg.strip().startswith("["):
                    import json as _json
                    vals = _json.loads(raw_cfg)
                    return [str(v).strip() for v in vals if str(v).strip()]
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                pass
            return [s.strip() for s in str(raw_cfg).split(',') if s.strip()]
        return default

    SANDBOX_DEFAULT_RUNTIME = _sbx_env_or_cfg("SANDBOX_DEFAULT_RUNTIME", "default_runtime", "docker").lower()
    SANDBOX_NETWORK_DEFAULT = _sbx_env_or_cfg("SANDBOX_NETWORK_DEFAULT", "network_default", "deny_all").lower()
    SANDBOX_MAX_UPLOAD_MB = _sbx_int("SANDBOX_MAX_UPLOAD_MB", "max_upload_mb", 64)
    SANDBOX_ARTIFACT_TTL_HOURS = _sbx_int("SANDBOX_ARTIFACT_TTL_HOURS", "artifact_ttl_hours", 24)
    SANDBOX_ARTIFACT_JANITOR_BACKGROUND_ENABLED = is_truthy(
        os.getenv("SANDBOX_ARTIFACT_JANITOR_BACKGROUND_ENABLED")
        or _sbx_get("artifact_janitor_background_enabled", "true")
        or "true"
    )
    SANDBOX_MAX_CONCURRENT_RUNS = _sbx_int("SANDBOX_MAX_CONCURRENT_RUNS", "max_concurrent_runs", 8)
    SANDBOX_ACTIVE_MAX_PER_USER = _sbx_int("SANDBOX_ACTIVE_MAX_PER_USER", "active_max_per_user", 0)
    SANDBOX_ACTIVE_MAX_PER_PERSONA = _sbx_int("SANDBOX_ACTIVE_MAX_PER_PERSONA", "active_max_per_persona", 0)
    SANDBOX_ACTIVE_MAX_PER_WORKSPACE = _sbx_int("SANDBOX_ACTIVE_MAX_PER_WORKSPACE", "active_max_per_workspace", 0)
    SANDBOX_ACTIVE_MAX_PER_WORKSPACE_GROUP = _sbx_int("SANDBOX_ACTIVE_MAX_PER_WORKSPACE_GROUP", "active_max_per_workspace_group", 0)
    SANDBOX_SESSION_DELETE_DRAIN_TIMEOUT_SEC = _sbx_int(
        "SANDBOX_SESSION_DELETE_DRAIN_TIMEOUT_SEC",
        "session_delete_drain_timeout_sec",
        10,
    )
    SANDBOX_ARTIFACT_JANITOR_INTERVAL_SEC = _sbx_int("SANDBOX_ARTIFACT_JANITOR_INTERVAL_SEC", "artifact_janitor_interval_sec", 30)
    SANDBOX_ARTIFACT_RECONCILE_INTERVAL_SEC = _sbx_int(
        "SANDBOX_ARTIFACT_RECONCILE_INTERVAL_SEC",
        "artifact_reconcile_interval_sec",
        300,
    )
    SANDBOX_SNAPSHOT_MAX_COUNT = _sbx_int("SANDBOX_SNAPSHOT_MAX_COUNT", "snapshot_max_count", 10)
    SANDBOX_SNAPSHOT_MAX_SIZE_MB = _sbx_int("SANDBOX_SNAPSHOT_MAX_SIZE_MB", "snapshot_max_size_mb", 256)
    SANDBOX_MAX_LOG_BYTES = _sbx_int("SANDBOX_MAX_LOG_BYTES", "max_log_bytes", 10 * 1024 * 1024)
    SANDBOX_PIDS_LIMIT = _sbx_int("SANDBOX_PIDS_LIMIT", "pids_limit", 256)
    SANDBOX_MAX_CPU = _sbx_float("SANDBOX_MAX_CPU", "max_cpu", 4.0)
    SANDBOX_MAX_MEM_MB = _sbx_int("SANDBOX_MAX_MEM_MB", "max_mem_mb", 8192)
    SANDBOX_WORKSPACE_CAP_MB = _sbx_int("SANDBOX_WORKSPACE_CAP_MB", "workspace_cap_mb", 256)
    # Queue/backpressure defaults for sandbox orchestrator
    SANDBOX_QUEUE_MAX_LENGTH = _sbx_int("SANDBOX_QUEUE_MAX_LENGTH", "queue_max_length", 100)
    SANDBOX_QUEUE_TTL_SEC = _sbx_int("SANDBOX_QUEUE_TTL_SEC", "queue_ttl_sec", 120)
    SANDBOX_QUEUE_MAX_PER_USER = _sbx_int("SANDBOX_QUEUE_MAX_PER_USER", "queue_max_per_user", 0)
    SANDBOX_QUEUE_MAX_PER_PERSONA = _sbx_int("SANDBOX_QUEUE_MAX_PER_PERSONA", "queue_max_per_persona", 0)
    SANDBOX_QUEUE_MAX_PER_WORKSPACE = _sbx_int("SANDBOX_QUEUE_MAX_PER_WORKSPACE", "queue_max_per_workspace", 0)
    SANDBOX_QUEUE_MAX_PER_WORKSPACE_GROUP = _sbx_int("SANDBOX_QUEUE_MAX_PER_WORKSPACE_GROUP", "queue_max_per_workspace_group", 0)
    SANDBOX_RUN_CLAIM_LEASE_SEC = _sbx_int("SANDBOX_RUN_CLAIM_LEASE_SEC", "run_claim_lease_sec", 30)
    # WebSocket server poll timeout (seconds) for sandbox log streams
    # Tests may override to a smaller value (e.g., 1) via env to speed disconnects
    SANDBOX_WS_POLL_TIMEOUT_SEC = _sbx_int("SANDBOX_WS_POLL_TIMEOUT_SEC", "ws_poll_timeout_sec", 30)
    SANDBOX_WS_MAX_CONNECTIONS_TOTAL = _sbx_int("SANDBOX_WS_MAX_CONNECTIONS_TOTAL", "ws_max_connections_total", 1024)
    SANDBOX_WS_MAX_CONNECTIONS_PER_USER = _sbx_int("SANDBOX_WS_MAX_CONNECTIONS_PER_USER", "ws_max_connections_per_user", 64)
    SANDBOX_WS_MAX_CONNECTIONS_PER_PERSONA = _sbx_int("SANDBOX_WS_MAX_CONNECTIONS_PER_PERSONA", "ws_max_connections_per_persona", 32)
    SANDBOX_WS_MAX_CONNECTIONS_PER_SESSION = _sbx_int("SANDBOX_WS_MAX_CONNECTIONS_PER_SESSION", "ws_max_connections_per_session", 16)
    SANDBOX_WS_MAX_CONNECTIONS_PER_RUN = _sbx_int("SANDBOX_WS_MAX_CONNECTIONS_PER_RUN", "ws_max_connections_per_run", 8)
    # Optional: signed WS URLs for log_stream_url issuance
    SANDBOX_WS_SIGNED_URL_TTL_SEC = _sbx_int("SANDBOX_WS_SIGNED_URL_TTL_SEC", "ws_signed_url_ttl_sec", 60)
    SANDBOX_WS_SIGNED_URLS = is_truthy(
        os.getenv("SANDBOX_WS_SIGNED_URLS")
        or _sbx_get("ws_signed_urls", "false")
        or "false"
    )
    SANDBOX_WS_SIGNING_SECRET = os.getenv("SANDBOX_WS_SIGNING_SECRET") or _sbx_get("ws_signing_secret", None)
    # Test-only helper: when true, the WS endpoint will publish synthetic
    # start/end frames to ensure subscribers see frames immediately. Disabled
    # by default; tests should set SANDBOX_WS_SYNTHETIC_FRAMES_FOR_TESTS=true.
    SANDBOX_WS_SYNTHETIC_FRAMES_FOR_TESTS = is_truthy(
        os.getenv("SANDBOX_WS_SYNTHETIC_FRAMES_FOR_TESTS")
        or _sbx_get("ws_synthetic_frames_for_tests", "false")
        or "false"
    )
    # Advertise spec 1.1 support by default (backward-compatible with 1.0)
    SANDBOX_SUPPORTED_SPEC_VERSIONS = _sbx_list("SANDBOX_SUPPORTED_SPEC_VERSIONS", "supported_spec_versions", ["1.0", "1.1"])
    SANDBOX_ENABLE_EXECUTION = is_truthy(
        os.getenv("SANDBOX_ENABLE_EXECUTION") or _sbx_get("enable_execution", "false") or "false"
    )
    SANDBOX_BACKGROUND_EXECUTION = is_truthy(
        os.getenv("SANDBOX_BACKGROUND_EXECUTION") or _sbx_get("background_execution", "false") or "false"
    )
    SANDBOX_IDEMPOTENCY_TTL_SEC = _sbx_int("SANDBOX_IDEMPOTENCY_TTL_SEC", "idempotency_ttl_sec", 600)
    # Default timeouts and cancel grace
    SANDBOX_DEFAULT_STARTUP_TIMEOUT_SEC = _sbx_int("SANDBOX_DEFAULT_STARTUP_TIMEOUT_SEC", "default_startup_timeout_sec", 20)
    SANDBOX_DEFAULT_EXEC_TIMEOUT_SEC = _sbx_int("SANDBOX_DEFAULT_EXEC_TIMEOUT_SEC", "default_exec_timeout_sec", 60)
    SANDBOX_CANCEL_GRACE_SECONDS = _sbx_int("SANDBOX_CANCEL_GRACE_SECONDS", "cancel_grace_seconds", 5)
    # Artifact caps (MB)
    SANDBOX_MAX_ARTIFACT_BYTES_PER_RUN_MB = _sbx_int("SANDBOX_MAX_ARTIFACT_BYTES_PER_RUN_MB", "max_artifact_bytes_per_run_mb", 32)
    SANDBOX_MAX_ARTIFACT_BYTES_PER_USER_MB = _sbx_int("SANDBOX_MAX_ARTIFACT_BYTES_PER_USER_MB", "max_artifact_bytes_per_user_mb", 128)
    # Security hardening knobs
    SANDBOX_ULIMIT_NOFILE = _sbx_int("SANDBOX_ULIMIT_NOFILE", "ulimit_nofile", 1024)
    SANDBOX_ULIMIT_NPROC = _sbx_int("SANDBOX_ULIMIT_NPROC", "ulimit_nproc", 512)
    # Optional: path to seccomp JSON and AppArmor profile name
    SANDBOX_DOCKER_SECCOMP = os.getenv("SANDBOX_DOCKER_SECCOMP") or _sbx_get("docker_seccomp", None)
    SANDBOX_DOCKER_APPARMOR_PROFILE = os.getenv("SANDBOX_DOCKER_APPARMOR_PROFILE") or _sbx_get("docker_apparmor_profile", None)
    # Store backend (sqlite|memory) and path
    # Default to in-memory store for MVP to align with PRD (can be overridden to 'sqlite')
    SANDBOX_STORE_BACKEND = _sbx_env_or_cfg("SANDBOX_STORE_BACKEND", "store_backend", "memory").lower()
    SANDBOX_STORE_DB_PATH = os.getenv("SANDBOX_STORE_DB_PATH") or _sbx_get("store_db_path", None)

    persona_config = load_comprehensive_config()

    def _persona_cfg_get(option: str) -> Optional[str]:
        try:
            if persona_config and persona_config.has_section("persona"):
                return persona_config.get("persona", option, fallback=None)
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            return None
        return None

    def _persona_bool(env_key: str, option: str, default: bool) -> bool:
        env_value = os.getenv(env_key)
        if env_value is not None:
            text = str(env_value).strip()
            if text and text.lower() not in {"none", "null", "nil"}:
                return is_truthy(text)
        cfg_value = _persona_cfg_get(option)
        if cfg_value is None:
            return default
        return is_truthy(str(cfg_value))

    def _persona_int(env_key: str, option: str, default: int) -> int:
        env_value = os.getenv(env_key)
        if env_value is not None:
            text = str(env_value).strip()
            if text and text.lower() not in {"none", "null", "nil"}:
                try:
                    return int(text)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{env_key} must be an integer") from exc

        cfg_value = _persona_cfg_get(option)
        if cfg_value is None:
            return default
        text = str(cfg_value).strip()
        if not text or text.lower() in {"none", "null", "nil"}:
            return default
        try:
            return int(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"[persona] {option} must be an integer") from exc

    PERSONA_DIALOGUE_TREE_EVAL_ENABLED = _persona_bool(
        "PERSONA_DIALOGUE_TREE_EVAL_ENABLED",
        "dialogue_tree_eval_enabled",
        False,
    )
    PERSONA_RUNTIME_EXPLORER_ENABLED = _persona_bool(
        "PERSONA_RUNTIME_EXPLORER_ENABLED",
        "runtime_explorer_enabled",
        False,
    )
    PERSONA_RUNTIME_EXPLORER_MAX_DEPTH = _persona_int(
        "PERSONA_RUNTIME_EXPLORER_MAX_DEPTH",
        "runtime_explorer_max_depth",
        1,
    )
    PERSONA_RUNTIME_EXPLORER_MAX_BRANCHING = _persona_int(
        "PERSONA_RUNTIME_EXPLORER_MAX_BRANCHING",
        "runtime_explorer_max_branching",
        2,
    )
    PERSONA_RUNTIME_EXPLORER_MAX_PROVIDER_CALLS = _persona_int(
        "PERSONA_RUNTIME_EXPLORER_MAX_PROVIDER_CALLS",
        "runtime_explorer_max_provider_calls",
        1,
    )
    PERSONA_RUNTIME_EXPLORER_TIMEOUT_MS = _persona_int(
        "PERSONA_RUNTIME_EXPLORER_TIMEOUT_MS",
        "runtime_explorer_timeout_ms",
        750,
    )
    PERSONA_RUNTIME_EXPLORER_MAX_TOKENS = _persona_int(
        "PERSONA_RUNTIME_EXPLORER_MAX_TOKENS",
        "runtime_explorer_max_tokens",
        256,
    )
    PERSONA_RUNTIME_EXPLORER_P95_ADDED_LATENCY_MS = _persona_int(
        "PERSONA_RUNTIME_EXPLORER_P95_ADDED_LATENCY_MS",
        "runtime_explorer_p95_added_latency_ms",
        1000,
    )
    PERSONA_RUNTIME_EXPLORER_LLM_JUDGES_ENABLED = _persona_bool(
        "PERSONA_RUNTIME_EXPLORER_LLM_JUDGES_ENABLED",
        "runtime_explorer_llm_judges_enabled",
        False,
    )
    PERSONA_DIALOGUE_TREE_TRACE_RETENTION_DAYS = _persona_int(
        "PERSONA_DIALOGUE_TREE_TRACE_RETENTION_DAYS",
        "dialogue_tree_trace_retention_days",
        7,
    )

    config_dict = {
        # General App
        "APP_MODE_STR": single_user_mode_str,
        "SINGLE_USER_MODE": single_user_mode,
        "LOG_LEVEL": log_level,
        "PROJECT_ROOT": ACTUAL_PROJECT_ROOT, # Centralized project root definition
        "CONTENT_DB_BACKEND": content_backend_mode,

        # Single User
        "SINGLE_USER_FIXED_ID": single_user_fixed_id,
        "SINGLE_USER_API_KEY": single_user_api_key,

        # Multi User / Auth
        "JWT_SECRET_KEY": jwt_secret_key,
        "JWT_ALGORITHM": jwt_algorithm,
        "ACCESS_TOKEN_EXPIRE_MINUTES": access_token_expire_minutes,
        "DATABASE_URL": database_url,
        "USER_DB_BASE_DIR": user_data_base_dir, # Renamed for clarity (was user_db_base_dir)
        "USERS_DB_CONFIGURED": users_db_configured,
        "SERVER_CLIENT_ID": SERVER_CLIENT_ID,

        # Redis Configuration
        "REDIS_HOST": redis_host,
        "REDIS_PORT": redis_port,
        "REDIS_DB": redis_db,
        "REDIS_URL": redis_url,
        "CACHE_TTL": cache_ttl,
        "REDIS_ENABLED": redis_enabled,
        # Character-Chat (rate limiting) - file-backed defaults; env may override later
        "CHARACTER_RATE_LIMIT_OPS": _character_rate_limit_ops,
        "CHARACTER_RATE_LIMIT_WINDOW": _character_rate_limit_window,
        "MAX_CHARACTERS_PER_USER": _max_characters_per_user,
        "MAX_CHARACTER_IMPORT_SIZE_MB": _max_character_import_size_mb,
        "MAX_CHATS_PER_USER": _max_chats_per_user,
        "MAX_MESSAGES_PER_CHAT": _max_messages_per_chat,
        "MAX_MESSAGES_PER_CHAT_SOFT": _max_messages_per_chat_soft,
        "MAX_CHAT_COMPLETIONS_PER_MINUTE": _max_chat_completions_per_minute,
        "MAX_MESSAGE_SENDS_PER_MINUTE": _max_message_sends_per_minute,

        # Chat Configuration - Load from config file with default
        "CHAT_DICT_MAX_TOKENS": int(
            comprehensive_config.get('Chat-Dictionaries', {}).get('max_tokens', '5000')
        ),

        # Web Scraping Configuration - Load from config file with default
        "STEALTH_WAIT_MS": int(
            comprehensive_config.get('Web-Scraping', {}).get('stealth_wait_ms', '5000')
        ),

        # Merge relevant parts from comprehensive_config
        # Embedding Config: support both 'embedding_config' and 'Embeddings' sections
        "EMBEDDING_CONFIG": (
            comprehensive_config.get("embedding_config")
            or comprehensive_config.get("Embeddings")
            or comprehensive_config.get("EMBEDDINGS")
            or {
                'embedding_provider': 'openai', # Fallback defaults
                'embedding_model': 'text-embedding-3-small',
                'onnx_model_path': "./Models/onnx_models/text-embedding-3-small.onnx",
                'model_dir': "./Models",
                'embedding_api_url': "http://localhost:8080/v1/embeddings",
                'embedding_api_key': '',
                'chunk_size': 400,
                'chunk_overlap': 200
            }
        ),
        # HuggingFace remote code allowlist (wildcards supported via fnmatch)
        # HuggingFace remote code allowlist (wildcards supported via fnmatch)
        # Precedence: ENV > config.txt (Embeddings.trusted_hf_remote_code_models) > default
        "TRUSTED_HF_REMOTE_CODE_MODELS": (lambda _env_val, _cfg: (
            [s.strip() for s in _env_val.split(",") if s.strip()] if _env_val is not None else (
                [s.strip() for s in str(_cfg).split(",") if s.strip()] if _cfg is not None else ["*stella*"]
            )
        ))(
            os.getenv("TRUSTED_HF_REMOTE_CODE_MODELS"),
            (
                (comprehensive_config.get('embedding_config') or {}).get('trusted_hf_remote_code_models')
                if isinstance(comprehensive_config, dict) else None
            )
        ),
        # Sandbox settings (code interpreter)
        "SANDBOX_DEFAULT_RUNTIME": SANDBOX_DEFAULT_RUNTIME,
        "SANDBOX_NETWORK_DEFAULT": SANDBOX_NETWORK_DEFAULT,
        "SANDBOX_MAX_UPLOAD_MB": SANDBOX_MAX_UPLOAD_MB,
        "SANDBOX_ARTIFACT_TTL_HOURS": SANDBOX_ARTIFACT_TTL_HOURS,
        "SANDBOX_ARTIFACT_JANITOR_BACKGROUND_ENABLED": SANDBOX_ARTIFACT_JANITOR_BACKGROUND_ENABLED,
        "SANDBOX_MAX_CONCURRENT_RUNS": SANDBOX_MAX_CONCURRENT_RUNS,
        "SANDBOX_ACTIVE_MAX_PER_USER": SANDBOX_ACTIVE_MAX_PER_USER,
        "SANDBOX_ACTIVE_MAX_PER_PERSONA": SANDBOX_ACTIVE_MAX_PER_PERSONA,
        "SANDBOX_ACTIVE_MAX_PER_WORKSPACE": SANDBOX_ACTIVE_MAX_PER_WORKSPACE,
        "SANDBOX_ACTIVE_MAX_PER_WORKSPACE_GROUP": SANDBOX_ACTIVE_MAX_PER_WORKSPACE_GROUP,
        "SANDBOX_SESSION_DELETE_DRAIN_TIMEOUT_SEC": SANDBOX_SESSION_DELETE_DRAIN_TIMEOUT_SEC,
        "SANDBOX_ARTIFACT_JANITOR_INTERVAL_SEC": SANDBOX_ARTIFACT_JANITOR_INTERVAL_SEC,
        "SANDBOX_ARTIFACT_RECONCILE_INTERVAL_SEC": SANDBOX_ARTIFACT_RECONCILE_INTERVAL_SEC,
        "SANDBOX_SNAPSHOT_MAX_COUNT": SANDBOX_SNAPSHOT_MAX_COUNT,
        "SANDBOX_SNAPSHOT_MAX_SIZE_MB": SANDBOX_SNAPSHOT_MAX_SIZE_MB,
        "SANDBOX_MAX_LOG_BYTES": SANDBOX_MAX_LOG_BYTES,
        "SANDBOX_PIDS_LIMIT": SANDBOX_PIDS_LIMIT,
        "SANDBOX_MAX_CPU": SANDBOX_MAX_CPU,
        "SANDBOX_MAX_MEM_MB": SANDBOX_MAX_MEM_MB,
        "SANDBOX_WORKSPACE_CAP_MB": SANDBOX_WORKSPACE_CAP_MB,
        "SANDBOX_SUPPORTED_SPEC_VERSIONS": SANDBOX_SUPPORTED_SPEC_VERSIONS,
        "SANDBOX_WS_POLL_TIMEOUT_SEC": SANDBOX_WS_POLL_TIMEOUT_SEC,
        "SANDBOX_WS_SIGNED_URL_TTL_SEC": SANDBOX_WS_SIGNED_URL_TTL_SEC,
        "SANDBOX_WS_SIGNED_URLS": SANDBOX_WS_SIGNED_URLS,
        # WebSocket stream settings for sandbox UI/logs
        "SANDBOX_WS_SYNTHETIC_FRAMES_FOR_TESTS": SANDBOX_WS_SYNTHETIC_FRAMES_FOR_TESTS,
        "SANDBOX_IDEMPOTENCY_TTL_SEC": SANDBOX_IDEMPOTENCY_TTL_SEC,
        "SANDBOX_DEFAULT_STARTUP_TIMEOUT_SEC": SANDBOX_DEFAULT_STARTUP_TIMEOUT_SEC,
        "SANDBOX_DEFAULT_EXEC_TIMEOUT_SEC": SANDBOX_DEFAULT_EXEC_TIMEOUT_SEC,
        "SANDBOX_CANCEL_GRACE_SECONDS": SANDBOX_CANCEL_GRACE_SECONDS,
        "SANDBOX_ENABLE_EXECUTION": SANDBOX_ENABLE_EXECUTION,

        # Notes/Auto-Title configuration
        # Enable LLM-backed title generation and choose default strategy
        # Precedence: ENV > config.txt [Notes] > defaults
        "NOTES_TITLE_LLM_ENABLED": (lambda _env, _cp: (
            # env override first
            is_truthy(_env) if _env is not None else (
                is_truthy(_cp.get('Notes', 'title_llm_enabled', fallback='false'))
                if _cp and hasattr(_cp, 'get') and _cp.has_section('Notes') else False
            )
        ))(
            os.getenv('NOTES_TITLE_LLM_ENABLED') or os.getenv('NOTE_TITLE_LLM_ENABLED'),
            load_comprehensive_config(),
        ),
        "NOTES_TITLE_DEFAULT_STRATEGY": (lambda _env, _cp: (
            _env.strip().lower() if isinstance(_env, str) else (
                str(_cp.get('Notes', 'title_default_strategy', fallback='heuristic')).strip().lower()
                if _cp and hasattr(_cp, 'get') and _cp.has_section('Notes') else 'heuristic'
            )
        ))(
            os.getenv('NOTES_TITLE_DEFAULT_STRATEGY') or os.getenv('NOTE_TITLE_DEFAULT_STRATEGY'),
            load_comprehensive_config(),
        ),
        "SANDBOX_BACKGROUND_EXECUTION": SANDBOX_BACKGROUND_EXECUTION,
        "SANDBOX_MAX_ARTIFACT_BYTES_PER_RUN_MB": SANDBOX_MAX_ARTIFACT_BYTES_PER_RUN_MB,
        "SANDBOX_MAX_ARTIFACT_BYTES_PER_USER_MB": SANDBOX_MAX_ARTIFACT_BYTES_PER_USER_MB,
        "SANDBOX_ULIMIT_NOFILE": SANDBOX_ULIMIT_NOFILE,
        "SANDBOX_ULIMIT_NPROC": SANDBOX_ULIMIT_NPROC,
        "SANDBOX_DOCKER_SECCOMP": SANDBOX_DOCKER_SECCOMP,
        "SANDBOX_DOCKER_APPARMOR_PROFILE": SANDBOX_DOCKER_APPARMOR_PROFILE,
        "SANDBOX_STORE_BACKEND": SANDBOX_STORE_BACKEND,
        "SANDBOX_STORE_DB_PATH": SANDBOX_STORE_DB_PATH,
        "SANDBOX_RUN_CLAIM_LEASE_SEC": SANDBOX_RUN_CLAIM_LEASE_SEC,
        # Email ingestion/search rollout controls
        "EMAIL_NATIVE_PERSIST_ENABLED": is_truthy(
            os.getenv("EMAIL_NATIVE_PERSIST_ENABLED", "true")
        ),
        "EMAIL_OPERATOR_SEARCH_ENABLED": is_truthy(
            os.getenv("EMAIL_OPERATOR_SEARCH_ENABLED", "true")
        ),
        "EMAIL_MEDIA_SEARCH_DELEGATION_MODE": (
            lambda raw_mode: (
                raw_mode if raw_mode in {"opt_in", "auto_email"} else "opt_in"
            )
        )(
            str(os.getenv("EMAIL_MEDIA_SEARCH_DELEGATION_MODE", "opt_in")).strip().lower()
        ),
        "GOVERNANCE_ROLLOUT_MODE": resolve_governance_rollout_mode(),
        "EMAIL_GMAIL_CONNECTOR_ENABLED": is_truthy(
            os.getenv("EMAIL_GMAIL_CONNECTOR_ENABLED", "false")
        ),
        # --- HYDE/doc2query (per-chunk) feature flags ---
        "HYDE_ENABLED": is_truthy(os.getenv("HYDE_ENABLED", "false")),
        "HYDE_QUESTIONS_PER_CHUNK": int(os.getenv("HYDE_QUESTIONS_PER_CHUNK", "0")),
        "HYDE_PROVIDER": os.getenv("HYDE_PROVIDER"),
        "HYDE_MODEL": os.getenv("HYDE_MODEL"),
        "HYDE_TEMPERATURE": float(os.getenv("HYDE_TEMPERATURE", "0.2")),
        "HYDE_MAX_TOKENS": int(os.getenv("HYDE_MAX_TOKENS", "96")),
        "HYDE_LANGUAGE": os.getenv("HYDE_LANGUAGE", "auto"),
        "HYDE_PROMPT_VERSION": int(os.getenv("HYDE_PROMPT_VERSION", "1")),
        # Retrieval side HYDE controls
        "HYDE_WEIGHT_QUESTION_MATCH": float(os.getenv("HYDE_WEIGHT_QUESTION_MATCH", "0.05")),
        "HYDE_K_FRACTION": float(os.getenv("HYDE_K_FRACTION", "0.5")),
        "HYDE_ONLY_IF_NEEDED": is_truthy(os.getenv("HYDE_ONLY_IF_NEEDED", "true")),
        "HYDE_SCORE_FLOOR": float(os.getenv("HYDE_SCORE_FLOOR", "0.30")),
        "HYDE_DEDUPE_BY_PARENT": is_truthy(os.getenv("HYDE_DEDUPE_BY_PARENT", "false")),
        # Add other configs from comprehensive_config as needed
        "OPENAI_API_KEY": comprehensive_config.get("openai_api", {}).get("api_key", os.getenv("OPENAI_API_KEY")),
        "TTS_HISTORY_ENABLED": tts_history_enabled,
        "TTS_HISTORY_STORE_TEXT": tts_history_store_text,
        "TTS_HISTORY_STORE_FAILED": tts_history_store_failed,
        "TTS_HISTORY_HASH_KEY": tts_history_hash_key,
        "TTS_HISTORY_RETENTION_DAYS": tts_history_retention_days,
        "TTS_HISTORY_MAX_ROWS_PER_USER": tts_history_max_rows_per_user,
        "TTS_HISTORY_PURGE_INTERVAL_HOURS": tts_history_purge_interval_hours,
        # You can continue to merge other specific keys or whole sections
        "COMPREHENSIVE_CONFIG_RAW": comprehensive_config, # Store the raw one if needed elsewhere
        # Audit export streaming threshold (opt-in via env/config.txt)
        "AUDIT_EXPORT_STREAM_AUTO_MAX_ROWS": audit_stream_auto_max_rows,
        "AUDIT_STORAGE_MODE": audit_storage_mode,
        "AUDIT_SHARED_DB_PATH": audit_shared_db_path,
        "AUDIT_STORAGE_ROLLBACK": audit_storage_rollback,
        "AUDIT_ETL_USER_SUBPATH": audit_etl_user_subpath,

        # Ephemeral cleanup worker (evals/rag pipeline ephemeral collections)
        "EPHEMERAL_CLEANUP_ENABLED": os.getenv("EPHEMERAL_CLEANUP_ENABLED", "false").lower() == "true",
        "EPHEMERAL_CLEANUP_INTERVAL_SEC": int(os.getenv("EPHEMERAL_CLEANUP_INTERVAL_SEC", "1800")),

        # Ingestion-time claims (factual statements) - env overrides config.txt [Claims]
        **(lambda: (
            (lambda _cp: (
                (lambda _env: (
                    {
                        "ENABLE_INGESTION_CLAIMS": (
                            (_env.get("ENABLE_INGESTION_CLAIMS").lower() == "true") if _env.get("ENABLE_INGESTION_CLAIMS") is not None else (
                                (_cp.getboolean('Claims', 'ENABLE_INGESTION_CLAIMS', fallback=False) if _cp else False)
                            )
                        ),
                        "CLAIM_EXTRACTOR_MODE": (
                            _env.get("CLAIM_EXTRACTOR_MODE") if _env.get("CLAIM_EXTRACTOR_MODE") is not None else (
                                _cp.get('Claims', 'CLAIM_EXTRACTOR_MODE', fallback='heuristic') if _cp else 'heuristic'
                            )
                        ),
                        "CLAIMS_MAX_PER_CHUNK": (
                            int(_env.get("CLAIMS_MAX_PER_CHUNK")) if _env.get("CLAIMS_MAX_PER_CHUNK") is not None else (
                                _cp.getint('Claims', 'CLAIMS_MAX_PER_CHUNK', fallback=3) if _cp else 3
                            )
                        ),
                        "CLAIMS_EMBED": (
                            (_env.get("CLAIMS_EMBED").lower() == "true") if _env.get("CLAIMS_EMBED") is not None else (
                                _cp.getboolean('Claims', 'CLAIMS_EMBED', fallback=False) if _cp else False
                            )
                        ),
                        "CLAIMS_EMBED_MODEL_ID": (
                            _env.get("CLAIMS_EMBED_MODEL_ID") if _env.get("CLAIMS_EMBED_MODEL_ID") is not None else (
                                _cp.get('Claims', 'CLAIMS_EMBED_MODEL_ID', fallback='') if _cp else ''
                            )
                        ),
                        "CLAIMS_CLUSTER_METHOD": (
                            _env.get("CLAIMS_CLUSTER_METHOD") if _env.get("CLAIMS_CLUSTER_METHOD") is not None else (
                                _cp.get('Claims', 'CLAIMS_CLUSTER_METHOD', fallback='embeddings') if _cp else 'embeddings'
                            )
                        ),
                        "CLAIMS_CLUSTER_SIMILARITY_THRESHOLD": (
                            float(_env.get("CLAIMS_CLUSTER_SIMILARITY_THRESHOLD")) if _env.get("CLAIMS_CLUSTER_SIMILARITY_THRESHOLD") is not None else (
                                float(_cp.get('Claims', 'CLAIMS_CLUSTER_SIMILARITY_THRESHOLD', fallback='0.85')) if _cp else 0.85
                            )
                        ),
                        "CLAIMS_CLUSTER_BATCH_SIZE": (
                            int(_env.get("CLAIMS_CLUSTER_BATCH_SIZE")) if _env.get("CLAIMS_CLUSTER_BATCH_SIZE") is not None else (
                                int(_cp.getint('Claims', 'CLAIMS_CLUSTER_BATCH_SIZE', fallback=200)) if _cp else 200
                            )
                        ),
                        # Claims LLM selection (provider + optional knobs)
                        "CLAIMS_LLM_PROVIDER": (
                            _env.get("CLAIMS_LLM_PROVIDER") if _env.get("CLAIMS_LLM_PROVIDER") is not None else (
                                _cp.get('Claims', 'CLAIMS_LLM_PROVIDER', fallback='') if _cp else ''
                            )
                        ),
                        "CLAIMS_LLM_MODEL": (
                            _env.get("CLAIMS_LLM_MODEL") if _env.get("CLAIMS_LLM_MODEL") is not None else (
                                _cp.get('Claims', 'CLAIMS_LLM_MODEL', fallback='') if _cp else ''
                            )
                        ),
                        "CLAIMS_LLM_TEMPERATURE": (
                            (float(_env.get("CLAIMS_LLM_TEMPERATURE")) if _env.get("CLAIMS_LLM_TEMPERATURE") is not None else (
                                float(_cp.get('Claims', 'CLAIMS_LLM_TEMPERATURE', fallback='0.1')) if _cp else 0.1
                            ))
                        ),
                        "CLAIMS_JOB_BUDGET_ENABLED": (
                            (_env.get("CLAIMS_JOB_BUDGET_ENABLED").lower() == "true") if _env.get("CLAIMS_JOB_BUDGET_ENABLED") is not None else (
                                _cp.getboolean('Claims', 'CLAIMS_JOB_BUDGET_ENABLED', fallback=False) if _cp else False
                            )
                        ),
                        "CLAIMS_JOB_MAX_COST_USD": (
                            float(_env.get("CLAIMS_JOB_MAX_COST_USD")) if _env.get("CLAIMS_JOB_MAX_COST_USD") is not None else (
                                float((_cp.get('Claims', 'CLAIMS_JOB_MAX_COST_USD', fallback='0') if _cp else '0') or 0)
                            )
                        ),
                        "CLAIMS_JOB_MAX_TOKENS": (
                            int(_env.get("CLAIMS_JOB_MAX_TOKENS")) if _env.get("CLAIMS_JOB_MAX_TOKENS") is not None else (
                                int((_cp.get('Claims', 'CLAIMS_JOB_MAX_TOKENS', fallback='0') if _cp else '0') or 0)
                            )
                        ),
                        "CLAIMS_JOB_BUDGET_STRICT": (
                            (_env.get("CLAIMS_JOB_BUDGET_STRICT").lower() == "true") if _env.get("CLAIMS_JOB_BUDGET_STRICT") is not None else (
                                _cp.getboolean('Claims', 'CLAIMS_JOB_BUDGET_STRICT', fallback=False) if _cp else False
                            )
                        ),
                        # Optional: allow local NER model name in config for users who want NER
                        "CLAIMS_LOCAL_NER_MODEL": (
                            _env.get("CLAIMS_LOCAL_NER_MODEL") if _env.get("CLAIMS_LOCAL_NER_MODEL") is not None else (
                                _cp.get('Claims', 'CLAIMS_LOCAL_NER_MODEL', fallback='en_core_web_sm') if _cp else 'en_core_web_sm'
                            )
                        ),
                        "CLAIMS_EXTRACTOR_LANGUAGE_DEFAULT": (
                            _env.get("CLAIMS_EXTRACTOR_LANGUAGE_DEFAULT")
                            if _env.get("CLAIMS_EXTRACTOR_LANGUAGE_DEFAULT") is not None else (
                                _cp.get('Claims', 'CLAIMS_EXTRACTOR_LANGUAGE_DEFAULT', fallback='en') if _cp else 'en'
                            )
                        ),
                        "CLAIMS_LOCAL_NER_MODEL_MAP": (
                            _env.get("CLAIMS_LOCAL_NER_MODEL_MAP")
                            if _env.get("CLAIMS_LOCAL_NER_MODEL_MAP") is not None else (
                                _cp.get('Claims', 'CLAIMS_LOCAL_NER_MODEL_MAP', fallback='') if _cp else ''
                            )
                        ),
                        "CLAIMS_PROMPT_VALIDATION_MODE": (
                            _env.get("CLAIMS_PROMPT_VALIDATION_MODE")
                            if _env.get("CLAIMS_PROMPT_VALIDATION_MODE") is not None else (
                                _cp.get('Claims', 'CLAIMS_PROMPT_VALIDATION_MODE', fallback='warning') if _cp else 'warning'
                            )
                        ),
                        "CLAIMS_PROMPT_VALIDATION_STRICT": (
                            (_env.get("CLAIMS_PROMPT_VALIDATION_STRICT").lower() == "true") if _env.get("CLAIMS_PROMPT_VALIDATION_STRICT") is not None else (
                                _cp.getboolean('Claims', 'CLAIMS_PROMPT_VALIDATION_STRICT', fallback=False) if _cp else False
                            )
                        ),
                        "CLAIMS_CONTEXT_WINDOW_CHARS": (
                            _safe_int(
                                _env.get("CLAIMS_CONTEXT_WINDOW_CHARS"),
                                int(_cp.getint('Claims', 'CLAIMS_CONTEXT_WINDOW_CHARS', fallback=0)) if _cp else 0,
                            )
                        ),
                        "CLAIMS_EXTRACTION_PASSES": (
                            _safe_int(
                                _env.get("CLAIMS_EXTRACTION_PASSES"),
                                int(_cp.getint('Claims', 'CLAIMS_EXTRACTION_PASSES', fallback=1)) if _cp else 1,
                            )
                        ),
                    }
                ))({
                    k: os.getenv(k) for k in [
                        "ENABLE_INGESTION_CLAIMS", "CLAIM_EXTRACTOR_MODE", "CLAIMS_MAX_PER_CHUNK",
                        "CLAIMS_EMBED", "CLAIMS_EMBED_MODEL_ID", "CLAIMS_CLUSTER_METHOD",
                        "CLAIMS_CLUSTER_SIMILARITY_THRESHOLD", "CLAIMS_CLUSTER_BATCH_SIZE",
                        "CLAIMS_LLM_PROVIDER", "CLAIMS_LLM_TEMPERATURE", "CLAIMS_JOB_BUDGET_ENABLED",
                        "CLAIMS_JOB_MAX_COST_USD", "CLAIMS_JOB_MAX_TOKENS", "CLAIMS_JOB_BUDGET_STRICT",
                        "CLAIMS_LOCAL_NER_MODEL", "CLAIMS_EXTRACTOR_LANGUAGE_DEFAULT", "CLAIMS_LOCAL_NER_MODEL_MAP",
                        "CLAIMS_PROMPT_VALIDATION_MODE", "CLAIMS_PROMPT_VALIDATION_STRICT",
                        "CLAIMS_CONTEXT_WINDOW_CHARS", "CLAIMS_EXTRACTION_PASSES"
                    ]
                })
            ))(load_comprehensive_config())
        ))(),

        # Claims periodic rebuild worker
        "CLAIMS_REBUILD_ENABLED": os.getenv("CLAIMS_REBUILD_ENABLED", "false").lower() == "true",
        "CLAIMS_REBUILD_INTERVAL_SEC": int(os.getenv("CLAIMS_REBUILD_INTERVAL_SEC", "3600")),
        # Policy: missing | all | stale (stale requires CLAIMS_STALE_DAYS)
        "CLAIMS_REBUILD_POLICY": os.getenv("CLAIMS_REBUILD_POLICY", "missing"),
        "CLAIMS_STALE_DAYS": int(os.getenv("CLAIMS_STALE_DAYS", "7")),

        # Claims monitoring (alerts/config)
        **(lambda: (
            (lambda _cp, _env: (
                (lambda raw_cost: (
                    {
                        "CLAIMS_MONITORING_ENABLED": (
                            (_env.get("CLAIMS_MONITORING_ENABLED").lower() == "true") if _env.get("CLAIMS_MONITORING_ENABLED") is not None else (
                                _cp.getboolean('ClaimsMonitoring', 'CLAIMS_MONITORING_ENABLED', fallback=False) if _cp else False
                            )
                        ),
                        "CLAIMS_ALERT_THRESHOLD_DEFAULT": (
                            float(_env.get("CLAIMS_ALERT_THRESHOLD_DEFAULT")) if _env.get("CLAIMS_ALERT_THRESHOLD_DEFAULT") is not None else (
                                float(_cp.get('ClaimsMonitoring', 'CLAIMS_ALERT_THRESHOLD_DEFAULT', fallback='0.2')) if _cp else 0.2
                            )
                        ),
                        "CLAIMS_REBUILD_MAX_QUEUE_ALERT": (
                            int(_env.get("CLAIMS_REBUILD_MAX_QUEUE_ALERT")) if _env.get("CLAIMS_REBUILD_MAX_QUEUE_ALERT") is not None else (
                                int(_cp.get('ClaimsMonitoring', 'CLAIMS_REBUILD_MAX_QUEUE_ALERT', fallback='1000')) if _cp else 1000
                            )
                        ),
                        "CLAIMS_REBUILD_HEARTBEAT_WARN_SEC": (
                            int(_env.get("CLAIMS_REBUILD_HEARTBEAT_WARN_SEC")) if _env.get("CLAIMS_REBUILD_HEARTBEAT_WARN_SEC") is not None else (
                                int(_cp.get('ClaimsMonitoring', 'CLAIMS_REBUILD_HEARTBEAT_WARN_SEC', fallback='600')) if _cp else 600
                            )
                        ),
                        "CLAIMS_ALERTS_SCHEDULER_ENABLED": (
                            (_env.get("CLAIMS_ALERTS_SCHEDULER_ENABLED").lower() == "true") if _env.get("CLAIMS_ALERTS_SCHEDULER_ENABLED") is not None else (
                                _cp.getboolean('ClaimsMonitoring', 'CLAIMS_ALERTS_SCHEDULER_ENABLED', fallback=False) if _cp else False
                            )
                        ),
                        "CLAIMS_ALERTS_EVAL_INTERVAL_SEC": (
                            int(_env.get("CLAIMS_ALERTS_EVAL_INTERVAL_SEC")) if _env.get("CLAIMS_ALERTS_EVAL_INTERVAL_SEC") is not None else (
                                int(_cp.get('ClaimsMonitoring', 'CLAIMS_ALERTS_EVAL_INTERVAL_SEC', fallback='300')) if _cp else 300
                            )
                        ),
                        "CLAIMS_ALERTS_WINDOW_SEC": (
                            int(_env.get("CLAIMS_ALERTS_WINDOW_SEC")) if _env.get("CLAIMS_ALERTS_WINDOW_SEC") is not None else (
                                int(_cp.get('ClaimsMonitoring', 'CLAIMS_ALERTS_WINDOW_SEC', fallback='3600')) if _cp else 3600
                            )
                        ),
                        "CLAIMS_ALERTS_BASELINE_SEC": (
                            int(_env.get("CLAIMS_ALERTS_BASELINE_SEC")) if _env.get("CLAIMS_ALERTS_BASELINE_SEC") is not None else (
                                int(_cp.get('ClaimsMonitoring', 'CLAIMS_ALERTS_BASELINE_SEC', fallback='86400')) if _cp else 86400
                            )
                        ),
                        "CLAIMS_ALERT_EMAIL_DIGEST_ENABLED": (
                            (_env.get("CLAIMS_ALERT_EMAIL_DIGEST_ENABLED").lower() == "true") if _env.get("CLAIMS_ALERT_EMAIL_DIGEST_ENABLED") is not None else (
                                _cp.getboolean('ClaimsMonitoring', 'CLAIMS_ALERT_EMAIL_DIGEST_ENABLED', fallback=False) if _cp else False
                            )
                        ),
                        "CLAIMS_ALERT_EMAIL_DIGEST_INTERVAL_SEC": (
                            int(_env.get("CLAIMS_ALERT_EMAIL_DIGEST_INTERVAL_SEC")) if _env.get("CLAIMS_ALERT_EMAIL_DIGEST_INTERVAL_SEC") is not None else (
                                int(_cp.get('ClaimsMonitoring', 'CLAIMS_ALERT_EMAIL_DIGEST_INTERVAL_SEC', fallback='86400')) if _cp else 86400
                            )
                        ),
                        "CLAIMS_ALERT_EMAIL_DIGEST_MAX_EVENTS": (
                            int(_env.get("CLAIMS_ALERT_EMAIL_DIGEST_MAX_EVENTS")) if _env.get("CLAIMS_ALERT_EMAIL_DIGEST_MAX_EVENTS") is not None else (
                                int(_cp.get('ClaimsMonitoring', 'CLAIMS_ALERT_EMAIL_DIGEST_MAX_EVENTS', fallback='500')) if _cp else 500
                            )
                        ),
                        "CLAIMS_ADAPTIVE_THROTTLE_ENABLED": (
                            (_env.get("CLAIMS_ADAPTIVE_THROTTLE_ENABLED").lower() == "true") if _env.get("CLAIMS_ADAPTIVE_THROTTLE_ENABLED") is not None else (
                                _cp.getboolean('ClaimsMonitoring', 'CLAIMS_ADAPTIVE_THROTTLE_ENABLED', fallback=False) if _cp else False
                            )
                        ),
                        "CLAIMS_ADAPTIVE_THROTTLE_LATENCY_MS": (
                            float(_env.get("CLAIMS_ADAPTIVE_THROTTLE_LATENCY_MS")) if _env.get("CLAIMS_ADAPTIVE_THROTTLE_LATENCY_MS") is not None else (
                                float((_cp.get('ClaimsMonitoring', 'CLAIMS_ADAPTIVE_THROTTLE_LATENCY_MS', fallback='0') if _cp else '0') or 0)
                            )
                        ),
                        "CLAIMS_ADAPTIVE_THROTTLE_ERROR_RATE": (
                            float(_env.get("CLAIMS_ADAPTIVE_THROTTLE_ERROR_RATE")) if _env.get("CLAIMS_ADAPTIVE_THROTTLE_ERROR_RATE") is not None else (
                                float((_cp.get('ClaimsMonitoring', 'CLAIMS_ADAPTIVE_THROTTLE_ERROR_RATE', fallback='0') if _cp else '0') or 0)
                            )
                        ),
                        "CLAIMS_ADAPTIVE_THROTTLE_BUDGET_RATIO": (
                            float(_env.get("CLAIMS_ADAPTIVE_THROTTLE_BUDGET_RATIO")) if _env.get("CLAIMS_ADAPTIVE_THROTTLE_BUDGET_RATIO") is not None else (
                                float((_cp.get('ClaimsMonitoring', 'CLAIMS_ADAPTIVE_THROTTLE_BUDGET_RATIO', fallback='0') if _cp else '0') or 0)
                            )
                        ),
                        "CLAIMS_PROVIDER_COST_MULTIPLIERS": _safe_json_dict(raw_cost),
                    }
                ))(
                    _env.get("CLAIMS_PROVIDER_COST_MULTIPLIERS") if _env.get("CLAIMS_PROVIDER_COST_MULTIPLIERS") is not None else (
                        _cp.get('ClaimsMonitoring', 'CLAIMS_PROVIDER_COST_MULTIPLIERS', fallback='') if _cp else ''
                    )
                )
            ))(load_comprehensive_config(), {
                k: os.getenv(k) for k in [
                    "CLAIMS_MONITORING_ENABLED",
                    "CLAIMS_ALERT_THRESHOLD_DEFAULT",
                    "CLAIMS_REBUILD_MAX_QUEUE_ALERT",
                    "CLAIMS_REBUILD_HEARTBEAT_WARN_SEC",
                    "CLAIMS_ALERTS_SCHEDULER_ENABLED",
                    "CLAIMS_ALERTS_EVAL_INTERVAL_SEC",
                    "CLAIMS_ALERTS_WINDOW_SEC",
                    "CLAIMS_ALERTS_BASELINE_SEC",
                    "CLAIMS_ALERT_EMAIL_DIGEST_ENABLED",
                    "CLAIMS_ALERT_EMAIL_DIGEST_INTERVAL_SEC",
                    "CLAIMS_ALERT_EMAIL_DIGEST_MAX_EVENTS",
                    "CLAIMS_ADAPTIVE_THROTTLE_ENABLED",
                    "CLAIMS_ADAPTIVE_THROTTLE_LATENCY_MS",
                    "CLAIMS_ADAPTIVE_THROTTLE_ERROR_RATE",
                    "CLAIMS_ADAPTIVE_THROTTLE_BUDGET_RATIO",
                    "CLAIMS_PROVIDER_COST_MULTIPLIERS",
                ]
            })
        ))(),

        # Contextual retrieval defaults (parent/siblings) - from env or config.txt [RAG] section
        "RAG_CONTEXTUAL_DEFAULTS": (lambda: (
            # Build contextual defaults from env first, then config.txt [RAG] section
            (lambda _envs, _cfg: {
                "include_parent_document": (
                    (_envs.get("RAG_INCLUDE_PARENT_DOCUMENT").lower() == "true") if _envs.get("RAG_INCLUDE_PARENT_DOCUMENT") is not None else (
                        str(_cfg.get('include_parent_document', 'false')).lower() == 'true'
                    )
                ),
                "parent_max_tokens": (
                    _int_env_or_cfg(
                        _envs.get("RAG_PARENT_MAX_TOKENS"),
                        _cfg.get("parent_max_tokens", "1200"),
                        1200,
                    )
                ),
                "include_sibling_chunks": (
                    (_envs.get("RAG_INCLUDE_SIBLING_CHUNKS").lower() == "true") if _envs.get("RAG_INCLUDE_SIBLING_CHUNKS") is not None else (
                        str(_cfg.get('include_sibling_chunks', 'false')).lower() == 'true'
                    )
                ),
                "sibling_window": (
                    _int_env_or_cfg(
                        _envs.get("RAG_SIBLING_WINDOW"),
                        _cfg.get("sibling_window", "1"),
                        1,
                    )
                ),
            })(
                {
                    "RAG_INCLUDE_PARENT_DOCUMENT": os.getenv("RAG_INCLUDE_PARENT_DOCUMENT"),
                    "RAG_PARENT_MAX_TOKENS": os.getenv("RAG_PARENT_MAX_TOKENS"),
                    "RAG_INCLUDE_SIBLING_CHUNKS": os.getenv("RAG_INCLUDE_SIBLING_CHUNKS"),
                    "RAG_SIBLING_WINDOW": os.getenv("RAG_SIBLING_WINDOW"),
                },
                (lambda _cp: (
                    (lambda d: d)(
                        {k: (_cp.get('RAG', k, fallback=None) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else None)
                         for k in ['include_parent_document', 'parent_max_tokens', 'include_sibling_chunks', 'sibling_window']}
                    )
                ))(load_comprehensive_config())
            )
        ))(),
        "IMPLICIT_FEEDBACK_ENABLED": implicit_feedback_enabled(default=True),

        # RAG FlashRank reranker cache/model configuration
        "RAG_FLASHRANK_CACHE_DIR": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'flashrank_cache_dir', fallback=None) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else None
            )
        ))(os.getenv('RAG_FLASHRANK_CACHE_DIR'), load_comprehensive_config()),
        "RAG_FLASHRANK_MODEL_NAME": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'flashrank_model_name', fallback=None) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else None
            )
        ))(os.getenv('RAG_FLASHRANK_MODEL_NAME'), load_comprehensive_config()),

        # RAG LLM reranker configuration (provider/model)
        "RAG_LLM_RERANKER_PROVIDER": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'llm_reranker_provider', fallback=None) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else None
            )
        ))(os.getenv('RAG_LLM_RERANKER_PROVIDER'), load_comprehensive_config()),
        "RAG_LLM_RERANKER_MODEL": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'llm_reranker_model', fallback=None) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else None
            )
        ))(os.getenv('RAG_LLM_RERANKER_MODEL'), load_comprehensive_config()),

        # RAG llama.cpp (GGUF) reranker configuration
        "RAG_LLAMA_RERANKER_BIN": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'llama_reranker_binary', fallback=None) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else None
            )
        ))(os.getenv('RAG_LLAMA_RERANKER_BIN'), load_comprehensive_config()),
        "RAG_LLAMA_RERANKER_MODEL": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'llama_reranker_model', fallback=None) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else None
            )
        ))(os.getenv('RAG_LLAMA_RERANKER_MODEL'), load_comprehensive_config()),
        "RAG_LLAMA_RERANKER_NGL": (lambda _env, _cp: (
            int(_env) if _env is not None else (
                int(_cp.get('RAG', 'llama_reranker_ngl', fallback='0')) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else 0
            )
        ))(os.getenv('RAG_LLAMA_RERANKER_NGL'), load_comprehensive_config()),
        "RAG_LLAMA_RERANKER_SEP": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'llama_reranker_separator', fallback='<#sep#>') if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else '<#sep#>'
            )
        ))(os.getenv('RAG_LLAMA_RERANKER_SEP'), load_comprehensive_config()),
        "RAG_LLAMA_RERANKER_OUTPUT": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'llama_reranker_output', fallback='json+') if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else 'json+'
            )
        ))(os.getenv('RAG_LLAMA_RERANKER_OUTPUT'), load_comprehensive_config()),
        "RAG_LLAMA_RERANKER_POOLING": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'llama_reranker_pooling', fallback='last') if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else 'last'
            )
        ))(os.getenv('RAG_LLAMA_RERANKER_POOLING'), load_comprehensive_config()),
        "RAG_LLAMA_RERANKER_NORMALIZE": (lambda _env, _cp: (
            int(_env) if _env is not None else (
                int(_cp.get('RAG', 'llama_reranker_normalize', fallback='-1')) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else -1
            )
        ))(os.getenv('RAG_LLAMA_RERANKER_NORMALIZE'), load_comprehensive_config()),
        "RAG_LLAMA_RERANKER_MAX_DOC_CHARS": (lambda _env, _cp: (
            int(_env) if _env is not None else (
                int(_cp.get('RAG', 'llama_reranker_max_doc_chars', fallback='2000')) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else 2000
            )
        ))(os.getenv('RAG_LLAMA_RERANKER_MAX_DOC_CHARS'), load_comprehensive_config()),
        "RAG_LLAMA_RERANKER_TEMPLATE_MODE": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'llama_reranker_template_mode', fallback='auto') if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else 'auto'
            )
        ))(os.getenv('RAG_LLAMA_RERANKER_TEMPLATE_MODE'), load_comprehensive_config()),
        "RAG_LLAMA_RERANKER_QUERY_PREFIX": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'llama_reranker_query_prefix', fallback=None) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else None
            )
        ))(os.getenv('RAG_LLAMA_RERANKER_QUERY_PREFIX'), load_comprehensive_config()),
        "RAG_LLAMA_RERANKER_DOC_PREFIX": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'llama_reranker_doc_prefix', fallback=None) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else None
            )
        ))(os.getenv('RAG_LLAMA_RERANKER_DOC_PREFIX'), load_comprehensive_config()),

        # Transformers cross-encoder reranker defaults
        "RAG_TRANSFORMERS_RERANKER_MODEL": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'transformers_reranker_model', fallback=None) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else None
            )
        ))(os.getenv('RAG_TRANSFORMERS_RERANKER_MODEL'), load_comprehensive_config()),

        # RAG HyDE configuration (provider/model)
        "RAG_HYDE_PROVIDER": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'hyde_provider', fallback=None) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else None
            )
        ))(os.getenv('RAG_HYDE_PROVIDER'), load_comprehensive_config()),
        "RAG_HYDE_MODEL": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'hyde_model', fallback=None) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else None
            )
        ))(os.getenv('RAG_HYDE_MODEL'), load_comprehensive_config()),

        # RAG default LLM (used for general lightweight tasks like query expansion/gap analysis)
        "RAG_DEFAULT_LLM_PROVIDER": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'default_llm_provider', fallback=None) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else None
            )
        ))(os.getenv('RAG_DEFAULT_LLM_PROVIDER'), load_comprehensive_config()),
        "RAG_DEFAULT_LLM_MODEL": (lambda _env, _cp: (
            _env if _env is not None else (
                _cp.get('RAG', 'default_llm_model', fallback=None) if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else None
            )
        ))(os.getenv('RAG_DEFAULT_LLM_MODEL'), load_comprehensive_config()),

        # RAG default FTS level ('media' or 'chunk')
        "RAG_DEFAULT_FTS_LEVEL": (lambda _env, _cp: (
            (_env.lower() if isinstance(_env, str) else None) if _env is not None else (
                _cp.get('RAG', 'default_fts_level', fallback='media').lower() if _cp and hasattr(_cp, 'get') and _cp.has_section('RAG') else 'media'
            )
        ))(os.getenv('RAG_DEFAULT_FTS_LEVEL'), load_comprehensive_config()),

        # --- Feature Flags: Personalization & Persona Agent ---
        # Personalization
        "PERSONALIZATION_ENABLED": (lambda _cp: (
            _cp.getboolean('personalization', 'enabled', fallback=True) if _cp and hasattr(_cp, 'has_section') and _cp.has_section('personalization') else True
        ))(load_comprehensive_config()),
        "PERSONALIZATION_ALPHA": (lambda _cp: (
            float(_cp.get('personalization', 'alpha', fallback='0.2')) if _cp and _cp.has_section('personalization') else 0.2
        ))(load_comprehensive_config()),
        "PERSONALIZATION_BETA": (lambda _cp: (
            float(_cp.get('personalization', 'beta', fallback='0.6')) if _cp and _cp.has_section('personalization') else 0.6
        ))(load_comprehensive_config()),
        "PERSONALIZATION_GAMMA": (lambda _cp: (
            float(_cp.get('personalization', 'gamma', fallback='0.2')) if _cp and _cp.has_section('personalization') else 0.2
        ))(load_comprehensive_config()),
        "PERSONALIZATION_RECENCY_HALF_LIFE_DAYS": (lambda _cp: (
            int(_cp.get('personalization', 'recency_half_life_days', fallback='14')) if _cp and _cp.has_section('personalization') else 14
        ))(load_comprehensive_config()),

        # Persona Agent and RBAC
        "PERSONA_ENABLED": (lambda _cp: (
            _cp.getboolean('persona', 'enabled', fallback=True) if _cp and hasattr(_cp, 'has_section') and _cp.has_section('persona') else True
        ))(load_comprehensive_config()),
        "PERSONA_DEFAULT_PERSONA": (lambda _cp: (
            _cp.get('persona', 'default_persona', fallback='Research Assistant') if _cp and _cp.has_section('persona') else 'Research Assistant'
        ))(load_comprehensive_config()),
        "PERSONA_VOICE": (
            lambda _cp: (
                _cp.get("persona", "voice", fallback="default") if _cp and _cp.has_section("persona") else "default"
            )
        )(load_comprehensive_config()),
        "PERSONA_STT": (
            lambda _cp: (
                _cp.get("persona", "stt", fallback="faster_whisper")
                if _cp and _cp.has_section("persona")
                else "faster_whisper"
            )
        )(load_comprehensive_config()),
        "PERSONA_MAX_TOOL_STEPS": (
            lambda _cp: (
                int(_cp.get("persona", "max_tool_steps", fallback="3")) if _cp and _cp.has_section("persona") else 3
            )
        )(load_comprehensive_config()),
        "PERSONA_MEMORY_READ_MODE": (
            lambda _env, _cp: (
                str(_env).strip().lower()
                if _env is not None
                else (
                    _cp.get("persona", "persona_memory_read_mode", fallback="legacy_only").strip().lower()
                    if _cp and _cp.has_section("persona")
                    else "legacy_only"
                )
            )
        )(os.getenv("PERSONA_MEMORY_READ_MODE"), load_comprehensive_config()),
        "PERSONA_MEMORY_WRITE_MODE": (
            lambda _env, _cp: (
                str(_env).strip().lower()
                if _env is not None
                else (
                    _cp.get("persona", "persona_memory_write_mode", fallback="legacy_only").strip().lower()
                    if _cp and _cp.has_section("persona")
                    else "legacy_only"
                )
            )
        )(os.getenv("PERSONA_MEMORY_WRITE_MODE"), load_comprehensive_config()),
        "PERSONA_DIALOGUE_TREE_EVAL_ENABLED": PERSONA_DIALOGUE_TREE_EVAL_ENABLED,
        "PERSONA_RUNTIME_EXPLORER_ENABLED": PERSONA_RUNTIME_EXPLORER_ENABLED,
        "PERSONA_RUNTIME_EXPLORER_MAX_DEPTH": PERSONA_RUNTIME_EXPLORER_MAX_DEPTH,
        "PERSONA_RUNTIME_EXPLORER_MAX_BRANCHING": PERSONA_RUNTIME_EXPLORER_MAX_BRANCHING,
        "PERSONA_RUNTIME_EXPLORER_MAX_PROVIDER_CALLS": PERSONA_RUNTIME_EXPLORER_MAX_PROVIDER_CALLS,
        "PERSONA_RUNTIME_EXPLORER_TIMEOUT_MS": PERSONA_RUNTIME_EXPLORER_TIMEOUT_MS,
        "PERSONA_RUNTIME_EXPLORER_MAX_TOKENS": PERSONA_RUNTIME_EXPLORER_MAX_TOKENS,
        "PERSONA_RUNTIME_EXPLORER_P95_ADDED_LATENCY_MS": PERSONA_RUNTIME_EXPLORER_P95_ADDED_LATENCY_MS,
        "PERSONA_RUNTIME_EXPLORER_LLM_JUDGES_ENABLED": PERSONA_RUNTIME_EXPLORER_LLM_JUDGES_ENABLED,
        "PERSONA_DIALOGUE_TREE_TRACE_RETENTION_DAYS": PERSONA_DIALOGUE_TREE_TRACE_RETENTION_DAYS,
        "PERSONA_RBAC_ALLOW_EXPORT": (
            lambda _cp: (
                _cp.getboolean("persona.rbac", "allow_export", fallback=False)
                if _cp and _cp.has_section("persona.rbac")
                else False
            )
        )(load_comprehensive_config()),
        "PERSONA_RBAC_ALLOW_DELETE": (
            lambda _cp: (
                _cp.getboolean("persona.rbac", "allow_delete", fallback=False)
                if _cp and _cp.has_section("persona.rbac")
                else False
            )
        )(load_comprehensive_config()),
    }
    # Only include explicit Character-Chat CHARACTER_RATE_LIMIT_ENABLED if present in config.txt
    try:
        if _has_char_rl_enabled:
            config_dict["CHARACTER_RATE_LIMIT_ENABLED"] = _char_rl_enabled_bool
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        pass

    # Surface provider-specific configuration blocks for orchestrator callers.
    provider_keys = [
        "llama_api",
        "ooba_api",
        "kobold_api",
        "tabby_api",
        "vllm_api",
        "ollama_api",
        "aphrodite_api",
        "custom_openai_api",
        "custom_openai_api_2",
    ]
    provider_keys.extend(
        custom_openai_section_name(number)
        for number in iter_custom_openai_provider_numbers(start=3)
    )
    for provider_key in provider_keys:
        provider_cfg = comprehensive_config.get(provider_key)
        if isinstance(provider_cfg, dict):
            config_dict[provider_key] = provider_cfg

    # Ensure a generic local-LLM configuration is exposed.
    existing_local_llm_cfg = comprehensive_config.get("local_llm")
    if isinstance(existing_local_llm_cfg, dict):
        config_dict["local_llm"] = existing_local_llm_cfg
    else:
        def _parse_float(value: Optional[str], default: Optional[float] = None) -> Optional[float]:
            if value is None or str(value).strip() == "":
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _parse_int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
            if value is None or str(value).strip() == "":
                return default
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _parse_bool(value: Optional[str], default: bool) -> bool:
            if value is None:
                return default
            return is_truthy(value)

        def _parse_json(value: Optional[str]):
            if value is None or str(value).strip() == "":
                return None
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        llm_defaults = comprehensive_config.get("llm_api_settings", {}) or {}
        config_dict["local_llm"] = {
            "api_ip": (
                os.getenv("LOCAL_LLM_API_URL")
                or os.getenv("LOCAL_LLM_API_BASE")
                or os.getenv("LOCAL_LLM_API_IP")
                or os.getenv("LOCAL_LLM_BASE_URL")
                or "http://127.0.0.1:8080/v1/chat/completions"
            ),
            "api_key": os.getenv("LOCAL_LLM_API_KEY"),
            "model": os.getenv("LOCAL_LLM_MODEL"),
            "temperature": _parse_float(os.getenv("LOCAL_LLM_TEMPERATURE"), 0.7),
            "streaming": _parse_bool(os.getenv("LOCAL_LLM_STREAMING"), False),
            "top_p": _parse_float(os.getenv("LOCAL_LLM_TOP_P")),
            "top_k": _parse_int(os.getenv("LOCAL_LLM_TOP_K")),
            "min_p": _parse_float(os.getenv("LOCAL_LLM_MIN_P")),
            "max_tokens": _parse_int(os.getenv("LOCAL_LLM_MAX_TOKENS"), 4096),
            "seed": os.getenv("LOCAL_LLM_SEED"),
            "stop": _parse_json(os.getenv("LOCAL_LLM_STOP")),
            "response_format": _parse_json(os.getenv("LOCAL_LLM_RESPONSE_FORMAT")),
            "n": _parse_int(os.getenv("LOCAL_LLM_N")),
            "logit_bias": _parse_json(os.getenv("LOCAL_LLM_LOGIT_BIAS")),
            "presence_penalty": _parse_float(os.getenv("LOCAL_LLM_PRESENCE_PENALTY")),
            "frequency_penalty": _parse_float(os.getenv("LOCAL_LLM_FREQUENCY_PENALTY")),
            "logprobs": _parse_bool(os.getenv("LOCAL_LLM_LOGPROBS"), False),
            "top_logprobs": _parse_int(os.getenv("LOCAL_LLM_TOP_LOGPROBS")),
            "tools": _parse_json(os.getenv("LOCAL_LLM_TOOLS")),
            "tool_choice": _parse_json(os.getenv("LOCAL_LLM_TOOL_CHOICE")),
            "user_identifier": os.getenv("LOCAL_LLM_USER"),
            "api_timeout": _parse_int(
                os.getenv("LOCAL_LLM_API_TIMEOUT"),
                _parse_int(llm_defaults.get("local_api_timeout"), 120) or 120,
            ),
            "api_retries": _parse_int(
                os.getenv("LOCAL_LLM_API_RETRIES"),
                _parse_int(llm_defaults.get("local_api_retries"), 1) or 1,
            ),
            "api_retry_delay": _parse_int(
                os.getenv("LOCAL_LLM_API_RETRY_DELAY"),
                _parse_int(llm_defaults.get("local_api_retry_delay"), 1) or 1,
            ),
            # When True, drop non-OpenAI fields from chat payload for strict servers
            "strict_openai_compat": _parse_bool(os.getenv("LOCAL_LLM_STRICT_OPENAI_COMPAT"), False),
        }

    # --- Warnings ---
    if config_dict["SINGLE_USER_MODE"] and not config_dict["SINGLE_USER_API_KEY"]:
        # In test contexts, downgrade this startup-time message to WARNING
        # to avoid failing tests that treat ERROR logs as failures. The
        # actual server startup paths still validate and will hard-fail
        # when SINGLE_USER_API_KEY is required.
        try:
            import sys as _sys
            _in_test = is_explicit_pytest_runtime() or is_test_mode() or ("pytest" in _sys.modules)
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            _in_test = False

        _msg = (
            "SINGLE_USER_API_KEY is not configured. The server will refuse to start in single-user mode.\n"
            "Run `python -m tldw_Server_API.app.core.AuthNZ.initialize` and generate secure keys, "
            "then set SINGLE_USER_API_KEY in your environment or .env file."
        )
        if _in_test:
            _log_warning(_msg)
        else:
            _log_error(_msg)
    if not config_dict["SINGLE_USER_MODE"] and config_dict["JWT_SECRET_KEY"] == "a_very_insecure_default_secret_key_for_dev_only":
        _log_critical("SECURITY WARNING: Using default JWT_SECRET_KEY in multi-user mode. Set a strong JWT_SECRET_KEY environment variable!")
    if not config_dict["SINGLE_USER_MODE"] and not config_dict["USERS_DB_CONFIGURED"]:
         _log_warning("Multi-user mode enabled (APP_MODE=multi), but USERS_DB_ENABLED is not 'true'. User authentication will likely fail.")

    # Create necessary directories if they don't exist
    # Ensure main SQLite database directory exists
    if config_dict["DATABASE_URL"].startswith("sqlite:///"):
        main_db_file_path_str = config_dict["DATABASE_URL"].replace("sqlite:///", "")
        # Path() can take a full file path string.
        main_db_file_path = Path(main_db_file_path_str)
        # Ensure the path is absolute if it was constructed from env var and relative
        if not main_db_file_path.is_absolute():
             main_db_file_path = ACTUAL_PROJECT_ROOT / main_db_file_path
        main_db_file_path.parent.mkdir(parents=True, exist_ok=True)
        _log_info(f"Ensured main SQLite database directory exists: {main_db_file_path.parent}")

    # Ensure USER_DB_BASE_DIR exists (base for user-specific SQLite and ChromaDB)
    config_dict["USER_DB_BASE_DIR"].mkdir(parents=True, exist_ok=True)
    _log_info(f"Ensured user data base directory exists: {config_dict['USER_DB_BASE_DIR']}")

    return config_dict


@lru_cache(maxsize=1)
def load_comprehensive_config():
    current_file_path = Path(__file__).resolve()
    # Correct project_root calculation:
    # __file__ is .../tldw_Server_API/app/core/config.py
    # .parent -> .../app/core
    # .parent.parent -> .../app
    # .parent.parent.parent -> .../tldw_Server_API (This is the project root)
    project_root = current_file_path.parent.parent.parent

    # Load .env/.ENV files if they exist (API keys should be here).
    # Prefer TLDW_ENV_FILE when set, then canonical repo paths before fallbacks.
    repo_root = project_root.parent
    candidate_env_paths = _candidate_env_paths(project_root, repo_root)
    loaded_any_env = False
    for p in candidate_env_paths:
        try:
            if p.exists():
                _log_info(f"Loading environment variables from: {str(p)}")
                load_dotenv(dotenv_path=str(p), override=False)
                loaded_any_env = True
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            # Continue trying other candidates
            pass
    if not loaded_any_env:
        _log_info(
            "No .env/.ENV file found in "
            f"{project_root}, {project_root / 'Config_Files'}, {repo_root}, or {repo_root / 'Config_Files'}; "
            "using config.txt and system env"
        )

    config_path_obj = resolve_config_file()
    _log_info(f"Attempting to load comprehensive config from: {str(config_path_obj)}")

    config_parser = _load_config_parser()

    _log_info(f"load_comprehensive_config(): Sections found in config: {config_parser.sections()}")

    # Propagate selected RAG rollout flags from config.txt into process env when unset.
    # This lets modules that consult os.getenv read file-backed defaults.
    try:
        if hasattr(config_parser, 'has_section') and config_parser.has_section('RAG'):
            # Helper to set env only if missing
            def _env_default(key: str, value: Optional[str]):
                if value is None:
                    return
                if os.getenv(key) is None:
                    os.environ[key] = str(value)

            # Structure index
            _env_default(
                'RAG_ENABLE_STRUCTURE_INDEX',
                config_parser.get('RAG', 'enable_structure_index', fallback='true')
            )
            # Strict extractive mode
            _env_default(
                'RAG_STRICT_EXTRACTIVE',
                config_parser.get('RAG', 'strict_extractive', fallback='false')
            )
            # Hard citations requirement
            _env_default(
                'RAG_REQUIRE_HARD_CITATIONS',
                config_parser.get('RAG', 'require_hard_citations', fallback='false')
            )
            # Low-confidence behavior
            _env_default(
                'RAG_LOW_CONFIDENCE_BEHAVIOR',
                config_parser.get('RAG', 'low_confidence_behavior', fallback='continue')
            )
            # Agentic cache backend + TTL
            _env_default(
                'RAG_AGENTIC_CACHE_BACKEND',
                config_parser.get('RAG', 'agentic_cache_backend', fallback='memory')
            )
            _env_default(
                'RAG_AGENTIC_CACHE_TTL_SEC',
                config_parser.get('RAG', 'agentic_cache_ttl_sec', fallback='600')
            )
            _env_default(
                'IMPLICIT_FEEDBACK_ENABLED',
                str(implicit_feedback_enabled(default=True, config_parser=config_parser)).lower()
            )

        # Search-Agent defaults used by unified RAG endpoint/router.
        if hasattr(config_parser, 'has_section') and config_parser.has_section('Search-Agent'):
            _env_default(
                'SEARCH_QUERY_CLASSIFICATION',
                config_parser.get('Search-Agent', 'search_query_classification', fallback='false')
            )
            _env_default(
                'SEARCH_DEFAULT_MODE',
                config_parser.get('Search-Agent', 'search_default_mode', fallback='balanced')
            )
            _env_default(
                'SEARCH_QUERY_REFORMULATION',
                config_parser.get('Search-Agent', 'search_query_reformulation', fallback='true')
            )
            _env_default(
                'SEARCH_RESEARCH_LOOP',
                config_parser.get('Search-Agent', 'search_research_loop', fallback='false')
            )
            _env_default(
                'SEARCH_DISCUSSIONS_ENABLED',
                config_parser.get('Search-Agent', 'search_discussions_enabled', fallback='false')
            )
            _env_default(
                'SEARCH_DISCUSSION_PLATFORMS',
                config_parser.get('Search-Agent', 'search_discussion_platforms', fallback='reddit,stackoverflow,hackernews')
            )
            _env_default(
                'SEARCH_PROGRESS_STREAMING',
                config_parser.get('Search-Agent', 'search_progress_streaming', fallback='false')
            )
            _env_default(
                'SEARCH_URL_SCRAPING',
                config_parser.get('Search-Agent', 'search_url_scraping', fallback='true')
            )
            _env_default(
                'SEARCH_CLASSIFIER_PROVIDER',
                config_parser.get('Search-Agent', 'search_classifier_provider', fallback=None)
            )
            _env_default(
                'SEARCH_CLASSIFIER_MODEL',
                config_parser.get('Search-Agent', 'search_classifier_model', fallback=None)
            )
            _env_default(
                'SEARCH_MAX_ITERATIONS_SPEED',
                config_parser.get('Search-Agent', 'search_max_iterations_speed', fallback='0')
            )
            _env_default(
                'SEARCH_MAX_ITERATIONS_BALANCED',
                config_parser.get('Search-Agent', 'search_max_iterations_balanced', fallback='0')
            )
            _env_default(
                'SEARCH_MAX_ITERATIONS_QUALITY',
                config_parser.get('Search-Agent', 'search_max_iterations_quality', fallback='0')
            )
    except _CONFIG_NONCRITICAL_EXCEPTIONS as _rag_env_err:
        _log_debug(f"RAG env propagation skipped: {_rag_env_err}")

    # Propagate Streaming flags from config.txt into process env when unset.
    try:
        def _env_default(name: str, value: Optional[str]):
            if value is None:
                return
            if os.getenv(name) is None:
                os.environ[name] = str(value)

        # Unified streaming switch (affects chat and selected SSE endpoints)
        if hasattr(config_parser, 'has_section') and config_parser.has_section('Streaming'):
            _env_default('STREAMS_UNIFIED', config_parser.get('Streaming', 'streams_unified', fallback=None))

        # Chat streaming channel maxsize (bound the in-memory channel used when queueing is enabled)
        maxsize_val: Optional[str] = None
        if hasattr(config_parser, 'has_section') and config_parser.has_section('Chat-Module'):
            try:
                maxsize_val = config_parser.get('Chat-Module', 'chat_stream_channel_maxsize', fallback=None)
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                maxsize_val = None
        if (not maxsize_val) and hasattr(config_parser, 'has_section') and config_parser.has_section('Streaming'):
            try:
                maxsize_val = config_parser.get('Streaming', 'chat_stream_channel_maxsize', fallback=None)
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                maxsize_val = None
        _env_default('CHAT_STREAM_CHANNEL_MAXSIZE', maxsize_val)

        # Chat streaming metadata injection (adds tldw_* IDs to SSE chunks)
        meta_val: Optional[str] = None
        if hasattr(config_parser, 'has_section') and config_parser.has_section('Chat-Module'):
            try:
                meta_val = config_parser.get('Chat-Module', 'chat_stream_include_metadata', fallback=None)
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                meta_val = None
        if (not meta_val) and hasattr(config_parser, 'has_section') and config_parser.has_section('Streaming'):
            try:
                meta_val = config_parser.get('Streaming', 'chat_stream_include_metadata', fallback=None)
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                meta_val = None
        _env_default('CHAT_STREAM_INCLUDE_METADATA', meta_val)
    except _CONFIG_NONCRITICAL_EXCEPTIONS as _stream_env_err:
        _log_debug(f"Streaming env propagation skipped: {_stream_env_err}")

    # Propagate HTTP client settings from config.txt into env (if unset)
    try:
        if hasattr(config_parser, 'has_section') and config_parser.has_section('HTTP'):
            def _env_default_http(name: str, opt: str):
                try:
                    v = config_parser.get('HTTP', opt, fallback=None)
                except _CONFIG_NONCRITICAL_EXCEPTIONS:
                    v = None
                if v is not None and os.getenv(name) is None:
                    os.environ[name] = str(v)

            # Core timeouts
            _env_default_http('HTTP_CONNECT_TIMEOUT', 'connect_timeout')
            _env_default_http('HTTP_READ_TIMEOUT', 'read_timeout')
            _env_default_http('HTTP_WRITE_TIMEOUT', 'write_timeout')
            _env_default_http('HTTP_POOL_TIMEOUT', 'pool_timeout')

            # Retries & backoff
            _env_default_http('HTTP_RETRY_ATTEMPTS', 'retry_attempts')
            _env_default_http('HTTP_BACKOFF_BASE_MS', 'backoff_base_ms')
            _env_default_http('HTTP_BACKOFF_CAP_S', 'backoff_cap_s')

            # Connection limits
            _env_default_http('HTTP_MAX_CONNECTIONS', 'max_connections')
            _env_default_http('HTTP_MAX_KEEPALIVE_CONNECTIONS', 'max_keepalive_connections')

            # trust_env and default UA
            _env_default_http('HTTP_TRUST_ENV', 'trust_env')
            _env_default_http('HTTP_DEFAULT_USER_AGENT', 'default_user_agent')

            # Optional JSON and transport flags
            _env_default_http('HTTP_JSON_MAX_BYTES', 'json_max_bytes')
            _env_default_http('HTTP3_ENABLED', 'http3_enabled')

            # Proxy allowlist
            _env_default_http('PROXY_ALLOWLIST', 'proxy_allowlist')

            # TLS enforcement and pins
            _env_default_http('HTTP_ENFORCE_TLS_MIN', 'enforce_tls_min_version')
            _env_default_http('HTTP_TLS_MIN_VERSION', 'tls_min_version')
            _env_default_http('HTTP_CERT_PINS', 'cert_pins')

            # Allow or disable following redirects globally for simple GET/HEAD fetches
            _env_default_http('HTTP_ALLOW_REDIRECTS', 'allow_redirects')
            # Maximum redirect hops before erroring (default inherited in http_client)
            try:
                v = config_parser.get('HTTP', 'max_redirects', fallback=None)
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                v = None
            if v is not None and os.getenv('HTTP_MAX_REDIRECTS') is None:
                os.environ['HTTP_MAX_REDIRECTS'] = str(v)
            # Cross-host redirects (default: disabled)
            _env_default_http('HTTP_ALLOW_CROSS_HOST_REDIRECTS', 'allow_cross_host_redirects')
            # Scheme downgrade (https -> http) (default: disabled)
            _env_default_http('HTTP_ALLOW_SCHEME_DOWNGRADE', 'allow_scheme_downgrade')
    except _CONFIG_NONCRITICAL_EXCEPTIONS as _http_env_err:
        _log_debug(f"HTTP env propagation skipped: {_http_env_err}")

    # Propagate system log file settings from config.txt into env (if unset)
    try:
        if hasattr(config_parser, 'has_section') and config_parser.has_section('Logging'):
            def _env_default_logging(name: str, opt: str):
                try:
                    v = config_parser.get('Logging', opt, fallback=None)
                except _CONFIG_NONCRITICAL_EXCEPTIONS:
                    v = None
                if v is not None and os.getenv(name) is None:
                    os.environ[name] = str(v)

            _env_default_logging('SYSTEM_LOG_FILE_PATH', 'system_log_file_path')
            _env_default_logging('SYSTEM_LOG_FILE_MAX_ENTRIES', 'system_log_file_max_entries')
    except _CONFIG_NONCRITICAL_EXCEPTIONS as _log_env_err:
        _log_debug(f"System log env propagation skipped: {_log_env_err}")

    # Propagate egress policy settings from config.txt into env (if unset)
    if hasattr(config_parser, 'has_section') and config_parser.has_section('Egress'):
        def _env_default_egress(name: str, opt: str):
            try:
                v = config_parser.get('Egress', opt, fallback=None)
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                v = None
            if v is None:
                return
            v_str = str(v).strip()
            if not v_str:
                return
            if os.getenv(name) is None:
                os.environ[name] = v_str

        # Global allow/deny lists
        _env_default_egress('EGRESS_ALLOWLIST', 'egress_allowlist')
        _env_default_egress('EGRESS_DENYLIST', 'egress_denylist')

        # Workflows-specific allow/deny overrides
        _env_default_egress('WORKFLOWS_EGRESS_ALLOWLIST', 'workflows_allowlist')
        _env_default_egress('WORKFLOWS_EGRESS_DENYLIST', 'workflows_denylist')

        # Port and profile controls
        _env_default_egress('WORKFLOWS_EGRESS_ALLOWED_PORTS', 'allowed_ports')
        _env_default_egress('WORKFLOWS_EGRESS_PROFILE', 'profile')

        # Private IP blocking toggle
        _env_default_egress('WORKFLOWS_EGRESS_BLOCK_PRIVATE', 'block_private')

    return config_parser


# ----------------------------
# RAG Toggle Helpers (bool/str/int)
# ----------------------------
def _as_bool(val: object, default: bool) -> bool:
    if val is None:
        return default
    s = str(val).strip().lower()
    if is_truthy(s):  # truthy
        return True
    if s in ("0", "false", "no", "off", "n"):
        return False
    return default


def rag_enable_structure_index(default: bool = True) -> bool:
    v = os.getenv("RAG_ENABLE_STRUCTURE_INDEX")
    if v is None:
        try:
            cp = load_comprehensive_config()
            v = cp.get("RAG", "enable_structure_index", fallback=str(default)) if cp else str(default)
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            v = str(default)
    return _as_bool(v, default)


def rag_strict_extractive(default: bool = False) -> bool:
    v = os.getenv("RAG_STRICT_EXTRACTIVE")
    if v is None:
        try:
            cp = load_comprehensive_config()
            v = cp.get("RAG", "strict_extractive", fallback=str(default)) if cp else str(default)
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            v = str(default)
    return _as_bool(v, default)


def rag_require_hard_citations(default: bool = False) -> bool:
    v = os.getenv("RAG_REQUIRE_HARD_CITATIONS")
    if v is None:
        try:
            cp = load_comprehensive_config()
            v = cp.get("RAG", "require_hard_citations", fallback=str(default)) if cp else str(default)
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            v = str(default)
    return _as_bool(v, default)


def rag_low_confidence_behavior(default: str = "continue") -> str:
    v = os.getenv("RAG_LOW_CONFIDENCE_BEHAVIOR")
    if v is None:
        try:
            cp = load_comprehensive_config()
            v = cp.get("RAG", "low_confidence_behavior", fallback=default) if cp else default
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            v = default
    s = str(v).strip().lower()
    return s if s in ("continue", "ask", "decline") else default


def web_outbound_policy_mode(default: str = "compat") -> str:
    """Return the web outbound-policy mode with env-over-config precedence.

    Accepted values are ``compat`` and ``strict``. Invalid or missing values
    fall back to ``default``.
    """
    v = os.getenv("WEB_OUTBOUND_POLICY_MODE")
    if v is None:
        try:
            cp = load_comprehensive_config()
            v = default
            if cp:
                has_section = getattr(cp, "has_section", None)
                for section_name in ("Web-Scraper", "Web-Scraping"):
                    if callable(has_section) and not has_section(section_name):
                        continue
                    candidate = cp.get(
                        section_name,
                        "web_outbound_policy_mode",
                        fallback=None,
                    )
                    if candidate is not None and str(candidate).strip():
                        v = candidate
                        break
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            v = default
    s = str(v).strip().lower()
    return s if s in ("compat", "strict") else default


def rag_agentic_cache_backend(default: str = "memory") -> str:
    v = os.getenv("RAG_AGENTIC_CACHE_BACKEND")
    if v is None:
        try:
            cp = load_comprehensive_config()
            v = cp.get("RAG", "agentic_cache_backend", fallback=default) if cp else default
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            v = default
    s = str(v).strip().lower()
    return s if s in ("memory", "sqlite") else default


def rag_agentic_cache_ttl_sec(default: int = 600) -> int:
    v = os.getenv("RAG_AGENTIC_CACHE_TTL_SEC")
    if v is None:
        try:
            cp = load_comprehensive_config()
            v = cp.get("RAG", "agentic_cache_ttl_sec", fallback=str(default)) if cp else str(default)
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            v = str(default)
    try:
        return max(1, int(str(v)))
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        return default


def implicit_feedback_enabled(
    default: bool = True,
    config_parser: Optional[configparser.ConfigParser] = None,
) -> bool:
    """
    Check if implicit feedback collection is enabled.

    Resolution order:
      1) Environment variable IMPLICIT_FEEDBACK_ENABLED
      2) config.txt [RAG] section implicit_feedback_enabled key
      3) Provided default (True unless overridden)

    Returns:
        bool: True if implicit feedback should be collected
    """
    v = os.getenv("IMPLICIT_FEEDBACK_ENABLED")
    if v is None:
        cp = config_parser
        if cp is None:
            try:
                cp = load_comprehensive_config()
            except (FileNotFoundError, configparser.Error) as exc:
                _log_debug(
                    f"implicit_feedback_enabled: config load failed, falling back to default={default}: {exc}"
                )
                cp = None
        if cp is not None:
            try:
                v = cp.get("RAG", "implicit_feedback_enabled", fallback=str(default))
            except (configparser.Error, AttributeError, TypeError, ValueError) as exc:
                _log_debug(
                    f"implicit_feedback_enabled: config read failed, falling back to default={default}: {exc}"
                )
                v = str(default)
        else:
            v = str(default)
    return _as_bool(v, default)


# ----------------------------
# Resource Governor Settings
# ----------------------------
def _as_int(val: object, default: int) -> int:
    try:
        return int(str(val))
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        return default


def rg_enabled(default: bool = True) -> bool:
    """
    Global feature flag for Resource Governor integrations.

    Resolution order:
      1) Env var RG_ENABLED
      2) [ResourceGovernor] enabled in config.txt
      3) Provided default (True unless overridden by caller)
    """
    v = os.getenv("RG_ENABLED")
    if v is None:
        # In test environments, keep RG disabled unless explicitly enabled via
        # RG_ENABLED so unit/integration tests don't unexpectedly start
        # receiving 429s when importing the main app.
        try:
            import sys as _sys

            _test_mode = is_test_mode()
            _pytest_active = is_explicit_pytest_runtime() or ("pytest" in _sys.modules)
            if _test_mode or _pytest_active:
                return False
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            cp = load_comprehensive_config()
        except (FileNotFoundError, configparser.Error) as exc:
            _log_debug(f"rg_enabled: config load failed, falling back to default={default}: {exc}")
            v = str(default)
        else:
            try:
                v = cp.get("ResourceGovernor", "enabled", fallback=str(default)) if cp else str(default)
            except (configparser.Error, KeyError, ValueError) as exc:
                _log_debug(f"rg_enabled: config read failed, falling back to default={default}: {exc}")
                v = str(default)
    return _as_bool(v, default)


def get_llamacpp_handler_config() -> Optional["LlamaCppConfig"]:
    """
    Build a LlamaCppConfig from environment variables or [LlamaCpp] section in config.txt.

    Env overrides (precedence over config.txt):
        LLAMACPP_ENABLED
        LLAMACPP_EXECUTABLE_PATH
        LLAMACPP_MODELS_DIR
        LLAMACPP_HOST
        LLAMACPP_PORT
        LLAMACPP_THREADS
        LLAMACPP_N_GPU_LAYERS
        LLAMACPP_CTX_SIZE
        LLAMACPP_ALLOW_UNVALIDATED_ARGS
        LLAMACPP_ALLOW_CLI_SECRETS
        LLAMACPP_PORT_AUTOSELECT
        LLAMACPP_PORT_PROBE_MAX
        LLAMACPP_ALLOWED_PATHS  (comma separated)
        LLAMACPP_LOG_OUTPUT_FILE
    """
    try:
        from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LlamaCppConfig
    except _CONFIG_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
        _log_warning(f"llama.cpp handler config unavailable (import failed): {exc}")
        return None

    try:
        cp = load_comprehensive_config()
    except _CONFIG_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
        _log_warning(f"llama.cpp handler config unavailable (config load failed): {exc}")
        return None
    section = cp["LlamaCpp"] if cp and cp.has_section("LlamaCpp") else None

    def _get(opt: str, env_name: str):
        v = os.getenv(env_name)
        if v is not None:
            return v
        if section is not None:
            try:
                return section.get(opt, fallback=None)
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                return None
        return None

    enabled_raw = _get("enabled", "LLAMACPP_ENABLED")
    enabled = _as_bool(enabled_raw, False)
    if not enabled:
        return None

    kwargs: dict[str, object] = {"enabled": True}

    exe = _get("executable_path", "LLAMACPP_EXECUTABLE_PATH")
    models_dir = _get("models_dir", "LLAMACPP_MODELS_DIR")
    default_host = _get("default_host", "LLAMACPP_HOST")
    default_port = _get("default_port", "LLAMACPP_PORT")
    default_threads = _get("default_threads", "LLAMACPP_THREADS")
    default_n_gpu_layers = _get("default_n_gpu_layers", "LLAMACPP_N_GPU_LAYERS")
    default_ctx_size = _get("default_ctx_size", "LLAMACPP_CTX_SIZE")
    allow_unvalidated_args = _get("allow_unvalidated_args", "LLAMACPP_ALLOW_UNVALIDATED_ARGS")
    allow_cli_secrets = _get("allow_cli_secrets", "LLAMACPP_ALLOW_CLI_SECRETS")
    port_autoselect = _get("port_autoselect", "LLAMACPP_PORT_AUTOSELECT")
    port_probe_max = _get("port_probe_max", "LLAMACPP_PORT_PROBE_MAX")
    allowed_paths_raw = _get("allowed_paths", "LLAMACPP_ALLOWED_PATHS")
    log_output_file = _get("log_output_file", "LLAMACPP_LOG_OUTPUT_FILE")

    if exe:
        kwargs["executable_path"] = Path(str(exe))
    if models_dir:
        kwargs["models_dir"] = Path(str(models_dir))
    if default_host:
        kwargs["default_host"] = str(default_host)
    if default_port is not None:
        kwargs["default_port"] = _as_int(default_port, 8080)
    if default_threads is not None and str(default_threads).strip():
        try:
            kwargs["default_threads"] = int(str(default_threads))
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            _log_warning(f"LLAMACPP_THREADS invalid; ignoring value: {default_threads}")
    if default_n_gpu_layers is not None:
        kwargs["default_n_gpu_layers"] = _as_int(default_n_gpu_layers, 0)
    if default_ctx_size is not None:
        kwargs["default_ctx_size"] = _as_int(default_ctx_size, 2048)
    if allow_unvalidated_args is not None:
        kwargs["allow_unvalidated_args"] = _as_bool(allow_unvalidated_args, False)
    if allow_cli_secrets is not None:
        kwargs["allow_cli_secrets"] = _as_bool(allow_cli_secrets, False)
    if port_autoselect is not None:
        kwargs["port_autoselect"] = _as_bool(port_autoselect, True)
    if port_probe_max is not None:
        kwargs["port_probe_max"] = _as_int(port_probe_max, 10)
    if allowed_paths_raw:
        paths = []
        for part in str(allowed_paths_raw).split(","):
            p = part.strip()
            if p:
                paths.append(Path(p))
        if paths:
            kwargs["allowed_paths"] = paths
    if log_output_file:
        kwargs["log_output_file"] = Path(str(log_output_file))

    try:
        return LlamaCppConfig(**kwargs)
    except _CONFIG_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
        _log_warning(f"llama.cpp handler config invalid; disabling handler: {exc}")
        return None


_GOVERNANCE_ROLLOUT_MODES = {"off", "shadow", "enforce"}
_RUN_FIRST_ROLLOUT_MODES = {"off", "gated", "default_on"}


def _split_run_first_provider_allowlist(raw: Optional[object]) -> list[str]:
    if raw is None:
        return []
    items: list[str] = []
    for chunk in str(raw).split(","):
        value = chunk.strip()
        if value:
            items.append(value)
    return items


def _resolve_run_first_rollout_mode(
    *,
    raw_mode: Optional[str],
    env_name: str,
    config_section: str,
    config_key: str,
    default: str,
) -> str:
    safe_default = str(default or "off").strip().lower()
    if safe_default not in _RUN_FIRST_ROLLOUT_MODES:
        safe_default = "off"

    if raw_mode is not None:
        candidate = str(raw_mode).strip().lower()
        return candidate if candidate in _RUN_FIRST_ROLLOUT_MODES else safe_default

    env_mode = os.getenv(env_name)
    if env_mode is not None:
        candidate = str(env_mode).strip().lower()
        return candidate if candidate in _RUN_FIRST_ROLLOUT_MODES else safe_default

    try:
        cp = load_comprehensive_config()
        if cp and cp.has_section(config_section):
            candidate = cp.get(config_section, config_key, fallback="").strip().lower()
            if candidate in _RUN_FIRST_ROLLOUT_MODES:
                return candidate
    except _CONFIG_NONCRITICAL_EXCEPTIONS as exc:
        _log_warning(
            f"_resolve_run_first_rollout_mode: unable to read "
            f"[{config_section}] {config_key}; falling back to '{safe_default}': {exc}"
        )

    return safe_default


def _resolve_run_first_provider_allowlist(
    *,
    raw_allowlist: Optional[str],
    env_name: str,
    config_section: str,
    config_key: str,
) -> list[str]:
    if raw_allowlist is not None:
        return _split_run_first_provider_allowlist(raw_allowlist)

    env_allowlist = os.getenv(env_name)
    if env_allowlist is not None:
        return _split_run_first_provider_allowlist(env_allowlist)

    try:
        cp = load_comprehensive_config()
        if cp and cp.has_section(config_section):
            return _split_run_first_provider_allowlist(
                cp.get(config_section, config_key, fallback="")
            )
    except _CONFIG_NONCRITICAL_EXCEPTIONS as exc:
        _log_warning(
            f"_resolve_run_first_provider_allowlist: unable to read "
            f"[{config_section}] {config_key}; falling back to empty allowlist: {exc}"
        )

    return []


def _resolve_run_first_presentation_variant(
    *,
    raw_variant: Optional[str],
    env_name: str,
    config_section: str,
    config_key: str,
    default: str,
) -> str:
    safe_default = str(default or "").strip()

    if raw_variant is not None:
        candidate = str(raw_variant).strip()
        return candidate or safe_default

    env_variant = os.getenv(env_name)
    if env_variant is not None:
        candidate = str(env_variant).strip()
        if candidate:
            return candidate

    try:
        cp = load_comprehensive_config()
        if cp and cp.has_section(config_section):
            candidate = cp.get(config_section, config_key, fallback="").strip()
            if candidate:
                return candidate
    except _CONFIG_NONCRITICAL_EXCEPTIONS as exc:
        _log_warning(
            f"_resolve_run_first_presentation_variant: unable to read "
            f"[{config_section}] {config_key}; falling back to '{safe_default}': {exc}"
        )

    return safe_default


def resolve_run_first_cohort_label(
    rollout_mode: str | None,
    *,
    eligible: bool,
    ineligible_reason: str | None = None,
) -> str:
    """Map run-first rollout state into the low-cardinality cohort label."""

    rollout_mode_name = str(rollout_mode or "").strip().lower()
    if rollout_mode_name == "gated":
        if (
            not eligible
            and str(ineligible_reason or "").strip() == "provider_not_in_rollout_allowlist"
        ):
            return "out_of_cohort"
        return "gated"
    if rollout_mode_name == "default_on":
        if eligible:
            return "default_on"
        if str(ineligible_reason or "").strip() == "provider_not_in_rollout_allowlist":
            return "out_of_cohort"
        return "default_on"
    return "override_off"
def resolve_chat_run_first_rollout_mode(
    raw_mode: Optional[str] = None,
    *,
    default: str = "off",
) -> str:
    """Resolve the chat run-first rollout mode from args, env, or config.

    Accepts the bounded rollout values supported by the shared resolver and
    falls back to ``default`` when the configured value is missing or invalid.
    """
    return _resolve_run_first_rollout_mode(
        raw_mode=raw_mode,
        env_name="CHAT_RUN_FIRST_ROLLOUT_MODE",
        config_section="Chat-Module",
        config_key="run_first_rollout_mode",
        default=default,
    )


def resolve_chat_run_first_provider_allowlist(
    raw_allowlist: Optional[str] = None,
) -> list[str]:
    """Resolve the chat run-first provider allowlist as normalized CSV values.

    Precedence is explicit arg, then environment variable, then config file.
    Empty or unreadable values resolve to an empty allowlist.
    """
    return _resolve_run_first_provider_allowlist(
        raw_allowlist=raw_allowlist,
        env_name="CHAT_RUN_FIRST_PROVIDER_ALLOWLIST",
        config_section="Chat-Module",
        config_key="run_first_provider_allowlist",
    )


def resolve_chat_run_first_presentation_variant(
    raw_variant: Optional[str] = None,
    *,
    default: str = "chat_phase2b_v1",
) -> str:
    """Resolve the chat run-first presentation variant.

    Uses explicit arg, env, and config precedence and falls back to the
    phase-2b default when the configured variant is blank or unreadable.
    """
    return _resolve_run_first_presentation_variant(
        raw_variant=raw_variant,
        env_name="CHAT_RUN_FIRST_PRESENTATION_VARIANT",
        config_section="Chat-Module",
        config_key="run_first_presentation_variant",
        default=default,
    )


def resolve_acp_run_first_rollout_mode(
    raw_mode: Optional[str] = None,
    *,
    default: str = "off",
) -> str:
    """Resolve the ACP run-first rollout mode from args, env, or config.

    Accepts the bounded rollout values supported by the shared resolver and
    falls back to ``default`` when the configured value is missing or invalid.
    """
    return _resolve_run_first_rollout_mode(
        raw_mode=raw_mode,
        env_name="ACP_RUN_FIRST_ROLLOUT_MODE",
        config_section="ACP",
        config_key="run_first_rollout_mode",
        default=default,
    )


def resolve_acp_run_first_provider_allowlist(
    raw_allowlist: Optional[str] = None,
) -> list[str]:
    """Resolve the ACP run-first provider allowlist as normalized CSV values.

    Precedence is explicit arg, then environment variable, then config file.
    Empty or unreadable values resolve to an empty allowlist.
    """
    return _resolve_run_first_provider_allowlist(
        raw_allowlist=raw_allowlist,
        env_name="ACP_RUN_FIRST_PROVIDER_ALLOWLIST",
        config_section="ACP",
        config_key="run_first_provider_allowlist",
    )


def resolve_acp_run_first_presentation_variant(
    raw_variant: Optional[str] = None,
    *,
    default: str = "acp_phase2b_v1",
) -> str:
    """Resolve the ACP run-first presentation variant.

    Uses explicit arg, env, and config precedence and falls back to the
    phase-2b default when the configured variant is blank or unreadable.
    """
    return _resolve_run_first_presentation_variant(
        raw_variant=raw_variant,
        env_name="ACP_RUN_FIRST_PRESENTATION_VARIANT",
        config_section="ACP",
        config_key="run_first_presentation_variant",
        default=default,
    )


def resolve_governance_rollout_mode(
    raw_mode: Optional[str] = None,
    *,
    default: str = "off",
) -> str:
    """
    Resolve governance rollout mode with deterministic fallback.

    Precedence:
    1) explicit `raw_mode` argument
    2) `GOVERNANCE_ROLLOUT_MODE` environment variable
    3) config.txt `[Governance] rollout_mode`
    4) provided default (fallbacks to `off` if invalid)
    """

    safe_default = str(default or "off").strip().lower()
    if safe_default not in _GOVERNANCE_ROLLOUT_MODES:
        safe_default = "off"

    if raw_mode is not None:
        candidate = str(raw_mode).strip().lower()
        return candidate if candidate in _GOVERNANCE_ROLLOUT_MODES else safe_default

    env_mode = os.getenv("GOVERNANCE_ROLLOUT_MODE")
    if env_mode is not None:
        candidate = str(env_mode).strip().lower()
        return candidate if candidate in _GOVERNANCE_ROLLOUT_MODES else safe_default

    try:
        cp = load_comprehensive_config()
        if cp and cp.has_section("Governance"):
            candidate = cp.get("Governance", "rollout_mode", fallback="").strip().lower()
            if candidate in _GOVERNANCE_ROLLOUT_MODES:
                return candidate
    except _CONFIG_NONCRITICAL_EXCEPTIONS as exc:
        _log_warning(
            f"resolve_governance_rollout_mode: unable to read [Governance] rollout_mode; "
            f"falling back to '{safe_default}': {exc}"
        )

    return safe_default


def rg_policy_store(default: str = "file") -> str:
    v = os.getenv("RG_POLICY_STORE")
    if v is None:
        try:
            cp = load_comprehensive_config()
            v = cp.get("ResourceGovernor", "policy_store", fallback=default) if cp else default
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            v = default
    s = str(v).strip().lower()
    return s if s in ("file", "db") else default


def rg_policy_reload_enabled(default: bool = True) -> bool:
    v = os.getenv("RG_POLICY_RELOAD_ENABLED")
    if v is None:
        try:
            cp = load_comprehensive_config()
            v = cp.get("ResourceGovernor", "policy_reload_enabled", fallback=str(default)) if cp else str(default)
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            v = str(default)
    return _as_bool(v, default)


def rg_policy_reload_interval_sec(default: int = 10) -> int:
    v = os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC")
    if v is None:
        try:
            cp = load_comprehensive_config()
            v = cp.get("ResourceGovernor", "policy_reload_interval_sec", fallback=str(default)) if cp else str(default)
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            v = str(default)
    return max(1, _as_int(v, default))


def rg_backend(default: str = "memory") -> str:
    v = os.getenv("RG_BACKEND")
    if v is None:
        try:
            cp = load_comprehensive_config()
            v = cp.get("ResourceGovernor", "backend", fallback=default) if cp else default
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            v = default
    s = str(v).strip().lower()
    return s if s in ("memory", "redis") else default


def rg_redis_fail_mode(default: str = "fallback_memory") -> str:
    v = os.getenv("RG_REDIS_FAIL_MODE")
    if v is None:
        try:
            cp = load_comprehensive_config()
            v = cp.get("ResourceGovernor", "redis_fail_mode", fallback=default) if cp else default
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            v = default
    s = str(v).strip().lower()
    return s if s in ("fail_closed", "fail_open", "fallback_memory") else default


def rg_repo_root() -> Path:
    """Return the repository root used for RG policy path resolution."""
    return Path(__file__).resolve().parents[3]


def resolve_repo_relative_path(path: str, base: Optional[Path] = None) -> str:
    """
    Resolve a path relative to the repository root when it is not absolute.

    Falls back to the original path if resolution fails, logging a debug message.
    """
    try:
        p = Path(path).expanduser()
        if not p.is_absolute():
            root = base or rg_repo_root()
            p = (root / p).resolve()
        return str(p)
    except (OSError, RuntimeError, ValueError) as exc:
        _log_debug(f"resolve_repo_relative_path: failed to resolve '{path}': {exc}")
        return path


def rg_policy_path_default() -> str:
    try:
        base = rg_repo_root()
        primary = base / "Config_Files" / "resource_governor_policies.yaml"
        if primary.exists():
            return str(primary)
        fallback = base / "tldw_Server_API" / "Config_Files" / "resource_governor_policies.yaml"
        return str(fallback)
    except (OSError, RuntimeError, ValueError) as exc:
        _log_debug(f"rg_policy_path_default: failed to resolve default policy path: {exc}")
        return "resource_governor_policies.yaml"


def rg_policy_path() -> str:
    v = os.getenv("RG_POLICY_PATH")
    if v:
        resolved = resolve_repo_relative_path(v)
        try:
            if not Path(resolved).exists():
                _log_warning(f"rg_policy_path: policy file not found at {resolved}")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            _log_debug(f"rg_policy_path: unable to check existence of {resolved}: {exc}")
        return resolved
    try:
        cp = load_comprehensive_config()
        p = cp.get("ResourceGovernor", "policy_path", fallback=rg_policy_path_default()) if cp else rg_policy_path_default()
        resolved = resolve_repo_relative_path(p)
        try:
            if not Path(resolved).exists():
                _log_warning(f"rg_policy_path: policy file not found at {resolved}")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            _log_debug(f"rg_policy_path: unable to check existence of {resolved}: {exc}")
        return resolved
    except (FileNotFoundError, configparser.Error, OSError, RuntimeError, ValueError) as exc:
        _log_debug(f"rg_policy_path: config load failed, using default: {exc}")
        return resolve_repo_relative_path(rg_policy_path_default())


@lru_cache(maxsize=1)
def should_disable_cors() -> bool:
    """Return True if CORS middleware should be skipped."""
    value, _ = _resolve_disable_cors_value_and_source()
    return value


@lru_cache(maxsize=1)
def should_allow_cors_credentials() -> bool:
    """Return True when CORS should allow credentials (cookies/auth)."""
    value, _ = _resolve_cors_allow_credentials_value_and_source()
    return value


@lru_cache(maxsize=1)
def is_production_environment() -> bool:
    """Return True when runtime environment should use production safety guards."""
    if is_truthy(os.getenv("tldw_production")):
        return True

    for env_key in ("ENV", "APP_ENV", "TLDW_ENV", "ENVIRONMENT"):
        raw_value = os.getenv(env_key)
        if raw_value is None:
            continue
        normalized = str(raw_value).strip().lower()
        if normalized in {"prod", "production"}:
            return True
    return False


def _resolve_disable_cors_value_and_source() -> tuple[bool, str]:
    env_value = os.getenv("DISABLE_CORS")
    if env_value is not None:
        return env_value.strip().lower() in {"true", "1", "yes", "on"}, "env"

    try:
        config_parser = _load_config_parser()
        if config_parser.has_section("Server"):
            has_option = config_parser.has_option("Server", "disable_cors")
            value = config_parser.getboolean("Server", "disable_cors", fallback=False)
            return value, ("config" if has_option else "default")
    except _CONFIG_NONCRITICAL_EXCEPTIONS as exc:
        _log_debug(f"Unable to read disable_cors flag from config: {exc}")

    return False, "default"


def _resolve_cors_allow_credentials_value_and_source() -> tuple[bool, str]:
    env_value = os.getenv("CORS_ALLOW_CREDENTIALS")
    if env_value is not None:
        return is_truthy(env_value), "env"

    try:
        config_parser = _load_config_parser()
        if config_parser.has_section("Server"):
            has_option = config_parser.has_option("Server", "cors_allow_credentials")
            value = config_parser.getboolean("Server", "cors_allow_credentials", fallback=False)
            return value, ("config" if has_option else "default")
    except _CONFIG_NONCRITICAL_EXCEPTIONS as exc:
        _log_debug(f"Unable to read cors_allow_credentials flag from config: {exc}")

    return False, "default"


def _resolve_allowed_origins_source() -> str:
    return _ALLOWED_ORIGINS_SOURCE


def resolve_runtime_allowed_origins(
    allowed_origins: list[str] | None = None,
) -> tuple[list[str], str, bool]:
    """Return effective CORS origins for the current runtime.

    Non-production startup falls back to the built-in local/loopback origins when
    configuration resolves to an explicit empty list so first-run self-hosting
    continues to work. Production keeps the empty list so startup validation can
    fail closed.
    """
    source = _resolve_allowed_origins_source()
    candidate_origins = ALLOWED_ORIGINS if allowed_origins is None else allowed_origins
    origins = [str(origin).strip() for origin in (candidate_origins or []) if str(origin).strip()]
    if origins:
        return origins, source, False

    if is_production_environment():
        return [], source, False

    return get_default_allowed_origins(), f"{source}(local-fallback)", True


def get_cors_runtime_diagnostics() -> dict[str, Any]:
    """Return effective CORS values with source attribution for startup diagnostics."""
    disable_cors, disable_cors_source = _resolve_disable_cors_value_and_source()
    allow_credentials, allow_credentials_source = _resolve_cors_allow_credentials_value_and_source()
    allowed_origins, allowed_origins_source, allowed_origins_fallback = resolve_runtime_allowed_origins(
        ALLOWED_ORIGINS
    )

    try:
        _load_config_parser()
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        pass
    config_meta = get_config_source_metadata()

    return {
        "disable_cors": disable_cors,
        "disable_cors_source": disable_cors_source,
        "allow_credentials": allow_credentials,
        "allow_credentials_source": allow_credentials_source,
        "allowed_origins": allowed_origins,
        "allowed_origins_count": len(allowed_origins),
        "allowed_origins_source": allowed_origins_source,
        "allowed_origins_fallback": allowed_origins_fallback,
        "config_path": config_meta.get("path"),
        "config_loaded": bool(config_meta.get("loaded")),
    }


def load_comprehensive_config_with_tts():
    """
    Load comprehensive configuration including TTS settings.

    Returns:
        Combined configuration object with TTS settings integrated
    """
    # Load main config
    config_parser = load_comprehensive_config()

    # Load TTS config
    tts_config = load_tts_config()

    # Create combined configuration object
    class CombinedConfig:
        def __init__(self, config_parser: configparser.ConfigParser, tts_config: dict[str, Any]):
            self.config_parser = config_parser
            self.tts_config = tts_config

        def get(self, section: str, key: str, fallback=None):
            """Get value from main config"""
            return self.config_parser.get(section, key, fallback=fallback)

        def has_section(self, section: str) -> bool:
            """Check if section exists in main config"""
            return self.config_parser.has_section(section)

        def has_option(self, section: str, key: str) -> bool:
            """Check if option exists in main config"""
            return self.config_parser.has_option(section, key)

        def items(self, section: str):
            """Get items from main config section"""
            return self.config_parser.items(section)

        def sections(self):
            """Get sections from main config"""
            return self.config_parser.sections()

        def get_tts_config(self) -> dict[str, Any]:
            """Get TTS configuration"""
            # Merge with API keys from main config for convenience
            merged_tts = self.tts_config.copy()

            # Add API keys if available
            if self.config_parser.has_option('API', 'openai_api_key'):
                merged_tts['openai_api_key'] = self.config_parser.get('API', 'openai_api_key')
            if self.config_parser.has_option('API', 'elevenlabs_api_key'):
                merged_tts['elevenlabs_api_key'] = self.config_parser.get('API', 'elevenlabs_api_key')

            return merged_tts

    return CombinedConfig(config_parser, tts_config)


# ----------------------------
# API Route Toggle Helpers
# ----------------------------

@lru_cache(maxsize=1)
def _route_toggle_policy() -> dict:
    """Compute route toggle policy from config.txt (authoritative in runtime).

    Priority:
      - Normal runtime: config.txt > defaults
      - Explicit pytest runtime: ENV overrides > config.txt > defaults

    Defaults:
      - Conservative baseline is `stable_only=True` when config cannot be read
        or the [API-Routes] section is missing.
      - If [API-Routes] exists but omits `stable_only`, parser fallback is false.
      - During explicit pytest runtime only, ROUTES_STABLE_ONLY can override.

    ENV variables supported (explicit pytest runtime only):
      - ROUTES_STABLE_ONLY: true|false
      - ROUTES_DISABLE: comma-separated list of route keys
      - ROUTES_ENABLE: comma-separated list of route keys
      - ROUTES_EXPERIMENTAL: comma-separated list of additional experimental route keys

    config.txt section [API-Routes] supports:
      stable_only = true|false
      disable = a,b,c
      enable = x,y,z
      experimental_routes = k1,k2  (optional, to extend defaults)
    """

    def _parse_list(val: Optional[str]) -> set[str]:
        if not val:
            return set()
        try:
            # Allow JSON array
            if val.strip().startswith("["):
                arr = json.loads(val)
                return {str(x).strip().lower() for x in arr if str(x).strip()}
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            pass
        # Comma/space separated
        parts = [p.strip() for p in val.replace("\n", ",").split(",")]
        return {p.lower() for p in parts if p}

    # Defaults: include all routes; a curated set flagged as experimental
    default_experimental: set[str] = {
        # Clearly in-development or optional scaffolds
        "sandbox",
        "connectors",
        "workflows",
        "scheduler",
        "personalization",
        "jobs",  # jobs admin
        "benchmarks",
        "guardian",
        "self-monitoring",
    }

    # Load from config.txt (if available). Keep a conservative default when
    # section/config are unavailable.
    cfg_stable_only = True
    cfg_disable: set[str] = set()
    cfg_enable: set[str] = set()
    cfg_experimental_extra: set[str] = set()
    try:
        cp = load_comprehensive_config()
        if cp and cp.has_section('API-Routes'):
            cfg_stable_only = cp.getboolean('API-Routes', 'stable_only', fallback=False)
            cfg_disable = _parse_list(cp.get('API-Routes', 'disable', fallback=''))
            cfg_enable = _parse_list(cp.get('API-Routes', 'enable', fallback=''))
            cfg_experimental_extra = _parse_list(cp.get('API-Routes', 'experimental_routes', fallback=''))
    except _CONFIG_NONCRITICAL_EXCEPTIONS as _e:
        _log_debug(f"Route policy: unable to read config.txt [API-Routes]: {_e}")

    # Keep config.txt authoritative in normal runtime.
    # Preserve env overrides only for explicit pytest execution where tests
    # rely on route toggling without mutating shared config files.
    allow_env_overrides = False
    try:
        # Route toggles are frequently consumed at import-time during test
        # collection, before PYTEST_CURRENT_TEST is set. Honor env overrides
        # whenever explicit pytest runtime is active OR server-side test mode
        # is enabled.
        allow_env_overrides = bool(is_explicit_pytest_runtime() or is_test_mode())
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        allow_env_overrides = False

    if allow_env_overrides:
        env_stable_only = os.getenv('ROUTES_STABLE_ONLY')
        if env_stable_only is not None:
            cfg_stable_only = is_truthy(env_stable_only)

        cfg_disable |= _parse_list(os.getenv('ROUTES_DISABLE'))
        cfg_enable |= _parse_list(os.getenv('ROUTES_ENABLE'))
        default_experimental |= _parse_list(os.getenv('ROUTES_EXPERIMENTAL'))
    default_experimental |= cfg_experimental_extra

    return {
        'stable_only': cfg_stable_only,
        'disable': cfg_disable,
        'enable': cfg_enable,
        'experimental': default_experimental,
    }


def route_enabled(route_key: str, *, default_stable: bool = True) -> bool:
    """Return True if a route identified by `route_key` should be included.

    - `route_key` should be a short, kebab/underscore case string used in main.py
    - If stable_only is True, routes marked experimental are disabled unless explicitly enabled
    - `disable` list always wins over implicit defaults; `enable` wins over stable_only gate
    - Unknown routes are treated as stable by default
    """
    key = (route_key or "").strip().lower()

    # Hard-enable critical test routes only in explicit pytest runtime.
    _minimal = env_flag_enabled("MINIMAL_TEST_APP")
    _pytest_runtime = is_explicit_pytest_runtime()
    if _minimal and _pytest_runtime and key in {"workflows", "scheduler"}:
        return True
    policy = _route_toggle_policy()

    # Expand aliases so a single key can control a family of routes.
    # Example: enabling "mcp" should enable both "mcp-unified" and "mcp-catalogs".
    try:
        enable = set(policy.get('enable', set()))
        disable = set(policy.get('disable', set()))
        if 'mcp' in enable:
            enable |= {'mcp-unified', 'mcp-catalogs'}
        if 'mcp' in disable:
            disable |= {'mcp-unified', 'mcp-catalogs'}
        # Reassign expanded sets for downstream checks
        policy = {**policy, 'enable': enable, 'disable': disable}
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        # On any unexpected structure, fall back to original policy
        pass

    # In test environments, force-enable certain routes commonly used by tests
    try:
        _test_mode = is_test_mode()
        _pytest_active = is_explicit_pytest_runtime()
        # Force-enable a small set of routes that tests rely on, regardless of
        # stable/experimental gating or import order. This avoids 404s when
        # the app module is imported before fixtures set ROUTES_ENABLE.
        _force_in_tests = {
            "workflows",
            "sandbox",
            "scheduler",
            "mcp-unified",
            "mcp-catalogs",
            "tools",
            "jobs",
            "scheduled-tasks",
            "personalization",
            "orgs",
            "org-invites",
            "prompt-studio",
            # Ensure experimental connectors endpoints are available in tests
            # to avoid 404s when the app is imported before ROUTES_ENABLE is set.
            "connectors",
            "llamacpp",
            "guardian",
            "self-monitoring",
        }
        if _pytest_active and _test_mode and key in _force_in_tests:
            return True
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        pass

    # Explicit allow/deny take precedence
    if key in policy['enable']:
        return True
    if key in policy['disable']:
        return False

    # Stable-only gate: exclude experimental by default when enabled
    if policy['stable_only'] and key in policy['experimental']:
        return False

    # Fallback default behavior: stable routes are enabled unless explicitly disabled
    return bool(default_stable)

def load_and_log_configs():
    _log_debug("load_and_log_configs(): Loading and logging configurations...")
    try:
        # The 'config' variable below should be the result from load_comprehensive_config()
        config_parser_object = load_comprehensive_config()

        # This check might be redundant if load_comprehensive_config always raises on critical failure
        if config_parser_object is None:
            _log_error("Comprehensive config object is None, cannot proceed")  # Changed to logger
            return None
        # API Keys - Check environment variables first, then config file
        anthropic_api_key = os.getenv('ANTHROPIC_API_KEY') or config_parser_object.get('API', 'anthropic_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Anthropic API Key: {anthropic_api_key[:5]}...{anthropic_api_key[-5:] if anthropic_api_key else None}")

        cohere_api_key = os.getenv('COHERE_API_KEY') or config_parser_object.get('API', 'cohere_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Cohere API Key: {cohere_api_key[:5]}...{cohere_api_key[-5:] if cohere_api_key else None}")

        groq_api_key = os.getenv('GROQ_API_KEY') or config_parser_object.get('API', 'groq_api_key', fallback=None)
        # logging.debug(f"Loaded Groq API Key: {groq_api_key[:5]}...{groq_api_key[-5:] if groq_api_key else None}")

        openai_api_key = os.getenv('OPENAI_API_KEY') or config_parser_object.get('API', 'openai_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded OpenAI API Key: {openai_api_key[:5]}...{openai_api_key[-5:] if openai_api_key else None}")

        huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY') or config_parser_object.get('API', 'huggingface_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded HuggingFace API Key: {huggingface_api_key[:5]}...{huggingface_api_key[-5:] if huggingface_api_key else None}")

        openrouter_api_key = os.getenv('OPENROUTER_API_KEY') or config_parser_object.get('API', 'openrouter_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded OpenRouter API Key: {openrouter_api_key[:5]}...{openrouter_api_key[-5:] if openrouter_api_key else None}")

        moonshot_api_key = os.getenv('MOONSHOT_API_KEY') or config_parser_object.get('API', 'moonshot_api_key', fallback=None)

        zai_api_key = os.getenv('ZAI_API_KEY') or config_parser_object.get('API', 'zai_api_key', fallback=None)

        deepseek_api_key = os.getenv('DEEPSEEK_API_KEY') or config_parser_object.get('API', 'deepseek_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded DeepSeek API Key: {deepseek_api_key[:5]}...{deepseek_api_key[-5:] if deepseek_api_key else None}")

        qwen_api_key = os.getenv('QWEN_API_KEY') or config_parser_object.get('API', 'qwen_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Qwen API Key: {qwen_api_key[:5]}...{qwen_api_key[-5:] if qwen_api_key else None}")

        mistral_api_key = os.getenv('MISTRAL_API_KEY') or config_parser_object.get('API', 'mistral_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Mistral API Key: {mistral_api_key[:5]}...{mistral_api_key[-5:] if mistral_api_key else None}")

        google_api_key = os.getenv('GOOGLE_API_KEY') or config_parser_object.get('API', 'google_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded Google API Key: {google_api_key[:5]}...{google_api_key[-5:] if google_api_key else None}")

        elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY') or config_parser_object.get('API', 'elevenlabs_api_key', fallback=None)
        # logging.debug(
        #     f"Loaded elevenlabs API Key: {elevenlabs_api_key[:5]}...{elevenlabs_api_key[-5:] if elevenlabs_api_key else None}")

        # LLM API Settings - streaming / temperature / top_p / min_p
        # Anthropic
        anthropic_model = config_parser_object.get('API', 'anthropic_model', fallback='claude-sonnet-4.5')
        anthropic_streaming = config_parser_object.get('API', 'anthropic_streaming', fallback='False')
        anthropic_temperature = config_parser_object.get('API', 'anthropic_temperature', fallback='0.7')
        anthropic_top_p = config_parser_object.get('API', 'anthropic_top_p', fallback='0.95')
        anthropic_top_k = config_parser_object.get('API', 'anthropic_top_k', fallback='100')
        anthropic_max_tokens = config_parser_object.get('API', 'anthropic_max_tokens', fallback='4096')
        anthropic_api_timeout = config_parser_object.get('API', 'anthropic_api_timeout', fallback='90')
        anthropic_api_retries = config_parser_object.get('API', 'anthropic_api_retry', fallback='3')
        anthropic_api_retry_delay = config_parser_object.get('API', 'anthropic_api_retry_delay', fallback='5')

        # Cohere
        cohere_streaming = config_parser_object.get('API', 'cohere_streaming', fallback='False')
        cohere_temperature = config_parser_object.get('API', 'cohere_temperature', fallback='0.7')
        cohere_max_p = config_parser_object.get('API', 'cohere_max_p', fallback='0.95')
        cohere_top_k = config_parser_object.get('API', 'cohere_top_k', fallback='100')
        cohere_model = config_parser_object.get('API', 'cohere_model', fallback='command-r-plus')
        cohere_max_tokens = config_parser_object.get('API', 'cohere_max_tokens', fallback='4096')
        cohere_api_timeout = config_parser_object.get('API', 'cohere_api_timeout', fallback='90')
        cohere_api_retries = config_parser_object.get('API', 'cohere_api_retry', fallback='3')
        cohere_api_retry_delay = config_parser_object.get('API', 'cohere_api_retry_delay', fallback='5')

        # Deepseek
        deepseek_streaming = config_parser_object.get('API', 'deepseek_streaming', fallback='False')
        deepseek_temperature = config_parser_object.get('API', 'deepseek_temperature', fallback='0.7')
        deepseek_top_p = config_parser_object.get('API', 'deepseek_top_p', fallback='0.95')
        deepseek_min_p = config_parser_object.get('API', 'deepseek_min_p', fallback='0.05')
        deepseek_model = config_parser_object.get('API', 'deepseek_model', fallback='deepseek-chat')
        deepseek_max_tokens = config_parser_object.get('API', 'deepseek_max_tokens', fallback='4096')
        deepseek_api_timeout = config_parser_object.get('API', 'deepseek_api_timeout', fallback='90')
        deepseek_api_retries = config_parser_object.get('API', 'deepseek_api_retry', fallback='3')
        deepseek_api_retry_delay = config_parser_object.get('API', 'deepseek_api_retry_delay', fallback='5')

        # Qwen (DashScope-compatible)
        qwen_model = config_parser_object.get('API', 'qwen_model', fallback='qwen-plus')
        qwen_streaming = config_parser_object.get('API', 'qwen_streaming', fallback='True')
        qwen_temperature = config_parser_object.get('API', 'qwen_temperature', fallback='0.7')
        qwen_top_p = config_parser_object.get('API', 'qwen_top_p', fallback='0.8')
        qwen_max_tokens = config_parser_object.get('API', 'qwen_max_tokens', fallback='4096')
        qwen_api_timeout = config_parser_object.get('API', 'qwen_api_timeout', fallback='90')
        qwen_api_retries = config_parser_object.get('API', 'qwen_api_retry', fallback='3')
        qwen_api_retry_delay = config_parser_object.get('API', 'qwen_api_retry_delay', fallback='1')
        qwen_api_region = config_parser_object.get('API', 'qwen_api_region', fallback='sg')
        qwen_api_base_url = config_parser_object.get(
            'API', 'qwen_api_base_url', fallback='https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
        )

        # Groq
        groq_model = config_parser_object.get('API', 'groq_model', fallback='llama3-70b-8192')
        groq_streaming = config_parser_object.get('API', 'groq_streaming', fallback='False')
        groq_temperature = config_parser_object.get('API', 'groq_temperature', fallback='0.7')
        groq_top_p = config_parser_object.get('API', 'groq_top_p', fallback='0.95')
        groq_max_tokens = config_parser_object.get('API', 'groq_max_tokens', fallback='4096')
        groq_api_timeout = config_parser_object.get('API', 'groq_api_timeout', fallback='90')
        groq_api_retries = config_parser_object.get('API', 'groq_api_retry', fallback='3')
        groq_api_retry_delay = config_parser_object.get('API', 'groq_api_retry_delay', fallback='5')

        # Google
        google_model = config_parser_object.get('API', 'google_model', fallback='gemini-2.5-flash')
        google_streaming = config_parser_object.get('API', 'google_streaming', fallback='False')
        google_temperature = config_parser_object.get('API', 'google_temperature', fallback='0.7')
        google_top_p = config_parser_object.get('API', 'google_top_p', fallback='0.95')
        google_min_p = config_parser_object.get('API', 'google_min_p', fallback='0.05')
        google_max_tokens = config_parser_object.get('API', 'google_max_tokens', fallback='4096')
        google_api_timeout = config_parser_object.get('API', 'google_api_timeout', fallback='90')
        google_api_retries = config_parser_object.get('API', 'google_api_retry', fallback='3')
        google_api_retry_delay = config_parser_object.get('API', 'google_api_retry_delay', fallback='5')

        # HuggingFace
        huggingface_use_router_url_format = config_parser_object.getboolean('API', 'huggingface_use_router_url_format', fallback=False)
        huggingface_router_base_url = config_parser_object.get('API', 'huggingface_router_base_url', fallback='https://router.huggingface.co/hf-inference')
        huggingface_api_base_url = config_parser_object.get('API', 'huggingface_api_base_url', fallback='https://router.huggingface.co/hf-inference/models')
        huggingface_api_chat_path = config_parser_object.get('API', 'huggingface_api_chat_path', fallback='v1/chat/completions')
        huggingface_model = config_parser_object.get('API', 'huggingface_model', fallback='/Qwen/Qwen3-235B-A22B')
        huggingface_streaming = config_parser_object.get('API', 'huggingface_streaming', fallback='False')
        huggingface_temperature = config_parser_object.get('API', 'huggingface_temperature', fallback='0.7')
        huggingface_top_p = config_parser_object.get('API', 'huggingface_top_p', fallback='0.95')
        huggingface_min_p = config_parser_object.get('API', 'huggingface_min_p', fallback='0.05')
        huggingface_max_tokens = config_parser_object.get('API', 'huggingface_max_tokens', fallback='4096')
        huggingface_api_timeout = config_parser_object.get('API', 'huggingface_api_timeout', fallback='90')
        huggingface_api_retries = config_parser_object.get('API', 'huggingface_api_retry', fallback='3')
        huggingface_api_retry_delay = config_parser_object.get('API', 'huggingface_api_retry_delay', fallback='5')

        # Mistral
        mistral_model = config_parser_object.get('API', 'mistral_model', fallback='mistral-large-latest')
        mistral_streaming = config_parser_object.get('API', 'mistral_streaming', fallback='False')
        mistral_temperature = config_parser_object.get('API', 'mistral_temperature', fallback='0.7')
        mistral_top_p = config_parser_object.get('API', 'mistral_top_p', fallback='0.95')
        mistral_max_tokens = config_parser_object.get('API', 'mistral_max_tokens', fallback='4096')
        mistral_api_timeout = config_parser_object.get('API', 'mistral_api_timeout', fallback='90')
        mistral_api_retries = config_parser_object.get('API', 'mistral_api_retry', fallback='3')
        mistral_api_retry_delay = config_parser_object.get('API', 'mistral_api_retry_delay', fallback='5')

        # OpenAI
        openai_model = config_parser_object.get('API', 'openai_model', fallback='gpt-4o')
        openai_streaming = config_parser_object.get('API', 'openai_streaming', fallback='False')
        openai_temperature = config_parser_object.get('API', 'openai_temperature', fallback='0.7')
        openai_top_p = config_parser_object.get('API', 'openai_top_p', fallback='0.95')
        openai_max_tokens = config_parser_object.get('API', 'openai_max_tokens', fallback='4096')
        openai_api_timeout = config_parser_object.get('API', 'openai_api_timeout', fallback='90')
        openai_api_retries = config_parser_object.get('API', 'openai_api_retry', fallback='3')
        openai_api_retry_delay = config_parser_object.get('API', 'openai_api_retry_delay', fallback='5')

        # OpenRouter
        openrouter_model = config_parser_object.get('API', 'openrouter_model', fallback='z-ai/glm-4.6')
        openrouter_streaming = config_parser_object.get('API', 'openrouter_streaming', fallback='False')
        openrouter_temperature = config_parser_object.get('API', 'openrouter_temperature', fallback='0.7')
        openrouter_top_p = config_parser_object.get('API', 'openrouter_top_p', fallback='0.95')
        openrouter_min_p = config_parser_object.get('API', 'openrouter_min_p', fallback='0.05')
        openrouter_top_k = config_parser_object.get('API', 'openrouter_top_k', fallback='100')
        openrouter_max_tokens = config_parser_object.get('API', 'openrouter_max_tokens', fallback='4096')
        openrouter_api_timeout = config_parser_object.get('API', 'openrouter_api_timeout', fallback='90')
        openrouter_api_retries = config_parser_object.get('API', 'openrouter_api_retry', fallback='3')
        openrouter_api_retry_delay = config_parser_object.get('API', 'openrouter_api_retry_delay', fallback='5')

        # Moonshot
        moonshot_model = config_parser_object.get('API', 'moonshot_model', fallback='moonshot-v1-8k')
        moonshot_api_base_url = config_parser_object.get('API', 'moonshot_api_base_url', fallback='https://api.moonshot.cn/v1')
        moonshot_streaming = config_parser_object.get('API', 'moonshot_streaming', fallback='False')
        moonshot_temperature = config_parser_object.get('API', 'moonshot_temperature', fallback='0.7')
        moonshot_top_p = config_parser_object.get('API', 'moonshot_top_p', fallback='0.95')
        moonshot_max_tokens = config_parser_object.get('API', 'moonshot_max_tokens', fallback='')
        moonshot_api_timeout = config_parser_object.get('API', 'moonshot_api_timeout', fallback='90')
        moonshot_api_retries = config_parser_object.get('API', 'moonshot_api_retry', fallback='3')
        moonshot_api_retry_delay = config_parser_object.get('API', 'moonshot_api_retry_delay', fallback='1')

        # Z.AI
        zai_model = config_parser_object.get('API', 'zai_model', fallback='glm-4.5-flash')
        zai_api_base_url = config_parser_object.get('API', 'zai_api_base_url', fallback='https://api.z.ai/api/paas/v4')
        zai_streaming = config_parser_object.get('API', 'zai_streaming', fallback='False')
        zai_temperature = config_parser_object.get('API', 'zai_temperature', fallback='0.7')
        zai_top_p = config_parser_object.get('API', 'zai_top_p', fallback='0.95')
        zai_max_tokens = config_parser_object.get('API', 'zai_max_tokens', fallback='4096')
        zai_api_timeout = config_parser_object.get('API', 'zai_api_timeout', fallback='90')
        zai_api_retries = config_parser_object.get('API', 'zai_api_retry', fallback='3')
        zai_api_retry_delay = config_parser_object.get('API', 'zai_api_retry_delay', fallback='1')

        # Bedrock
        bedrock_api_key = os.getenv('BEDROCK_API_KEY') or os.getenv('AWS_BEARER_TOKEN_BEDROCK') or config_parser_object.get('API', 'bedrock_api_key', fallback=None)
        bedrock_region = os.getenv('BEDROCK_REGION') or config_parser_object.get('API', 'bedrock_region', fallback='us-west-2')
        bedrock_runtime_endpoint = os.getenv('BEDROCK_RUNTIME_ENDPOINT') or config_parser_object.get('API', 'bedrock_runtime_endpoint', fallback=None)
        bedrock_model = os.getenv('BEDROCK_MODEL') or config_parser_object.get('API', 'bedrock_model', fallback=None)
        bedrock_streaming = config_parser_object.get('API', 'bedrock_streaming', fallback='False')
        bedrock_temperature = config_parser_object.get('API', 'bedrock_temperature', fallback='0.7')
        bedrock_top_p = config_parser_object.get('API', 'bedrock_top_p', fallback='')
        bedrock_max_tokens = config_parser_object.get('API', 'bedrock_max_tokens', fallback='')
        bedrock_api_timeout = config_parser_object.get('API', 'bedrock_api_timeout', fallback='90')
        bedrock_api_retries = config_parser_object.get('API', 'bedrock_api_retry', fallback='3')
        bedrock_api_retry_delay = config_parser_object.get('API', 'bedrock_api_retry_delay', fallback='5')

        # Logging Checks for model loads
        # logging.debug(f"Loaded Anthropic Model: {anthropic_model}")
        # logging.debug(f"Loaded Cohere Model: {cohere_model}")
        # logging.debug(f"Loaded Groq Model: {groq_model}")
        # logging.debug(f"Loaded OpenAI Model: {openai_model}")
        # logging.debug(f"Loaded HuggingFace Model: {huggingface_model}")
        # logging.debug(f"Loaded OpenRouter Model: {openrouter_model}")
        # logging.debug(f"Loaded Deepseek Model: {deepseek_model}")
        # logging.debug(f"Loaded Mistral Model: {mistral_model}")

        # Local-Models
        kobold_api_ip = config_parser_object.get('Local-API', 'kobold_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
        kobold_openai_api_IP = config_parser_object.get('Local-API', 'kobold_openai_api_IP', fallback='http://127.0.0.1:5001/v1/chat/completions')
        kobold_api_key = config_parser_object.get('Local-API', 'kobold_api_key', fallback='')
        kobold_streaming = config_parser_object.get('Local-API', 'kobold_streaming', fallback='False')
        kobold_temperature = config_parser_object.get('Local-API', 'kobold_temperature', fallback='0.7')
        kobold_top_p = config_parser_object.get('Local-API', 'kobold_top_p', fallback='0.95')
        kobold_top_k = config_parser_object.get('Local-API', 'kobold_top_k', fallback='100')
        kobold_max_tokens = config_parser_object.get('Local-API', 'kobold_max_tokens', fallback='4096')
        kobold_api_timeout = config_parser_object.get('Local-API', 'kobold_api_timeout', fallback='90')
        kobold_api_retries = config_parser_object.get('Local-API', 'kobold_api_retry', fallback='3')
        kobold_api_retry_delay = config_parser_object.get('Local-API', 'kobold_api_retry_delay', fallback='5')

        llama_api_IP = config_parser_object.get('Local-API', 'llama_api_IP', fallback='http://127.0.0.1:8080/v1/chat/completions')
        llama_api_key = config_parser_object.get('Local-API', 'llama_api_key', fallback='')
        llama_streaming = config_parser_object.get('Local-API', 'llama_streaming', fallback='False')
        llama_temperature = config_parser_object.get('Local-API', 'llama_temperature', fallback='0.7')
        llama_top_p = config_parser_object.get('Local-API', 'llama_top_p', fallback='0.95')
        llama_min_p = config_parser_object.get('Local-API', 'llama_min_p', fallback='0.05')
        llama_top_k = config_parser_object.get('Local-API', 'llama_top_k', fallback='100')
        llama_max_tokens = config_parser_object.get('Local-API', 'llama_max_tokens', fallback='4096')
        llama_api_timeout = config_parser_object.get('Local-API', 'llama_api_timeout', fallback='90')
        llama_api_retries = config_parser_object.get('Local-API', 'llama_api_retry', fallback='3')
        llama_api_retry_delay = config_parser_object.get('Local-API', 'llama_api_retry_delay', fallback='5')

        ooba_api_IP = config_parser_object.get('Local-API', 'ooba_api_IP', fallback='http://127.0.0.1:5000/v1/chat/completions')
        ooba_api_key = config_parser_object.get('Local-API', 'ooba_api_key', fallback='')
        ooba_streaming = config_parser_object.get('Local-API', 'ooba_streaming', fallback='False')
        ooba_temperature = config_parser_object.get('Local-API', 'ooba_temperature', fallback='0.7')
        ooba_top_p = config_parser_object.get('Local-API', 'ooba_top_p', fallback='0.95')
        ooba_min_p = config_parser_object.get('Local-API', 'ooba_min_p', fallback='0.05')
        ooba_top_k = config_parser_object.get('Local-API', 'ooba_top_k', fallback='100')
        ooba_max_tokens = config_parser_object.get('Local-API', 'ooba_max_tokens', fallback='4096')
        ooba_api_timeout = config_parser_object.get('Local-API', 'ooba_api_timeout', fallback='90')
        ooba_api_retries = config_parser_object.get('Local-API', 'ooba_api_retry', fallback='3')
        ooba_api_retry_delay = config_parser_object.get('Local-API', 'ooba_api_retry_delay', fallback='5')

        tabby_api_IP = config_parser_object.get('Local-API', 'tabby_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
        tabby_api_key = config_parser_object.get('Local-API', 'tabby_api_key', fallback=None)
        tabby_model = config_parser_object.get(
            'Local-API',
            'tabby_model',
            fallback=config_parser_object.get('models', 'tabby_model', fallback=None),
        )
        tabby_streaming = config_parser_object.get('Local-API', 'tabby_streaming', fallback='False')
        tabby_temperature = config_parser_object.get('Local-API', 'tabby_temperature', fallback='0.7')
        tabby_top_p = config_parser_object.get('Local-API', 'tabby_top_p', fallback='0.95')
        tabby_top_k = config_parser_object.get('Local-API', 'tabby_top_k', fallback='100')
        tabby_min_p = config_parser_object.get('Local-API', 'tabby_min_p', fallback='0.05')
        tabby_max_tokens = config_parser_object.get('Local-API', 'tabby_max_tokens', fallback='4096')
        tabby_api_timeout = config_parser_object.get('Local-API', 'tabby_api_timeout', fallback='90')
        tabby_api_retries = config_parser_object.get('Local-API', 'tabby_api_retry', fallback='3')
        tabby_api_retry_delay = config_parser_object.get('Local-API', 'tabby_api_retry_delay', fallback='5')

        vllm_api_url = config_parser_object.get('Local-API', 'vllm_api_IP', fallback='http://127.0.0.1:500/api/v1/chat/completions')
        vllm_api_key = config_parser_object.get('Local-API', 'vllm_api_key', fallback=None)
        vllm_model = config_parser_object.get('Local-API', 'vllm_model', fallback=None)
        vllm_streaming = config_parser_object.get('Local-API', 'vllm_streaming', fallback='False')
        vllm_temperature = config_parser_object.get('Local-API', 'vllm_temperature', fallback='0.7')
        vllm_top_p = config_parser_object.get('Local-API', 'vllm_top_p', fallback='0.95')
        vllm_top_k = config_parser_object.get('Local-API', 'vllm_top_k', fallback='100')
        vllm_min_p = config_parser_object.get('Local-API', 'vllm_min_p', fallback='0.05')
        vllm_max_tokens = config_parser_object.get('Local-API', 'vllm_max_tokens', fallback='4096')
        vllm_api_timeout = config_parser_object.get('Local-API', 'vllm_api_timeout', fallback='90')
        vllm_api_retries = config_parser_object.get('Local-API', 'vllm_api_retry', fallback='3')
        vllm_api_retry_delay = config_parser_object.get('Local-API', 'vllm_api_retry_delay', fallback='5')

        ollama_api_url = config_parser_object.get('Local-API', 'ollama_api_IP', fallback='http://127.0.0.1:11434/api/generate')
        ollama_api_key = config_parser_object.get('Local-API', 'ollama_api_key', fallback=None)
        ollama_model = config_parser_object.get('Local-API', 'ollama_model', fallback=None)
        ollama_streaming = config_parser_object.get('Local-API', 'ollama_streaming', fallback='False')
        ollama_temperature = config_parser_object.get('Local-API', 'ollama_temperature', fallback='0.7')
        ollama_top_p = config_parser_object.get('Local-API', 'ollama_top_p', fallback='0.95')
        ollama_max_tokens = config_parser_object.get('Local-API', 'ollama_max_tokens', fallback='4096')
        ollama_api_timeout = config_parser_object.get('Local-API', 'ollama_api_timeout', fallback='90')
        ollama_api_retries = config_parser_object.get('Local-API', 'ollama_api_retry', fallback='3')
        ollama_api_retry_delay = config_parser_object.get('Local-API', 'ollama_api_retry_delay', fallback='5')

        aphrodite_api_url = config_parser_object.get('Local-API', 'aphrodite_api_IP', fallback='http://127.0.0.1:8080/v1/chat/completions')
        aphrodite_api_key = config_parser_object.get('Local-API', 'aphrodite_api_key', fallback='')
        aphrodite_model = config_parser_object.get('Local-API', 'aphrodite_model', fallback='')
        aphrodite_max_tokens = config_parser_object.get('Local-API', 'aphrodite_max_tokens', fallback='4096')
        aphrodite_streaming = config_parser_object.get('Local-API', 'aphrodite_streaming', fallback='False')
        aphrodite_api_timeout = config_parser_object.get('Local-API', 'aphrodite_api_timeout', fallback='90')
        aphrodite_api_retries = config_parser_object.get('Local-API', 'aphrodite_api_retry', fallback='3')
        aphrodite_api_retry_delay = config_parser_object.get('Local-API', 'aphrodite_api_retry_delay', fallback='5')

        custom_openai_api_key = _env_or_config_option_value(
            custom_openai_api_key_env_keys(1),
            config_parser_object,
            'API',
            custom_openai_config_option_names(1, 'key'),
            fallback=None,
        )
        custom_openai_api_ip = _env_or_config_value(
            CUSTOM_OPENAI_ENDPOINT_ENV_KEYS,
            config_parser_object,
            'API',
            'custom_openai_api_ip',
            fallback=None,
        )
        custom_openai_api_model = _env_or_config_option_value(
            custom_openai_model_env_keys(1),
            config_parser_object,
            'API',
            custom_openai_config_option_names(1, 'model'),
            fallback=None,
        )
        custom_openai_api_streaming = config_parser_object.get('API', 'custom_openai_api_streaming', fallback='False')
        custom_openai_api_temperature = config_parser_object.get('API', 'custom_openai_api_temperature', fallback='0.7')
        custom_openai_api_top_p = config_parser_object.get('API', 'custom_openai_api_top_p', fallback='0.95')
        custom_openai_api_min_p = config_parser_object.get('API', 'custom_openai_api_min_p', fallback='0.05')
        custom_openai_api_max_tokens = config_parser_object.get('API', 'custom_openai_api_max_tokens', fallback='4096')
        custom_openai_api_timeout = config_parser_object.get('API', 'custom_openai_api_timeout', fallback='90')
        custom_openai_api_retries = config_parser_object.get('API', 'custom_openai_api_retry', fallback='3')
        custom_openai_api_retry_delay = config_parser_object.get('API', 'custom_openai_api_retry_delay', fallback='5')

        # 2nd Custom OpenAI API
        custom_openai2_api_key = _env_or_config_option_value(
            custom_openai_api_key_env_keys(2),
            config_parser_object,
            'API',
            custom_openai_config_option_names(2, 'key'),
            fallback=None,
        )
        custom_openai2_api_ip = _env_or_config_option_value(
            CUSTOM_OPENAI2_ENDPOINT_ENV_KEYS,
            config_parser_object,
            'API',
            custom_openai_config_option_names(2, 'ip'),
            fallback=None,
        )
        custom_openai2_api_model = _env_or_config_option_value(
            custom_openai_model_env_keys(2),
            config_parser_object,
            'API',
            custom_openai_config_option_names(2, 'model'),
            fallback=None,
        )
        custom_openai2_api_streaming = config_parser_object.get('API', 'custom_openai2_api_streaming', fallback='False')
        custom_openai2_api_temperature = config_parser_object.get('API', 'custom_openai2_api_temperature', fallback='0.7')
        custom_openai2_api_top_p = config_parser_object.get('API', 'custom_openai2_api_top_p', fallback='0.95')
        custom_openai2_api_min_p = config_parser_object.get('API', 'custom_openai2_api_min_p', fallback='0.05')
        custom_openai2_api_max_tokens = config_parser_object.get('API', 'custom_openai2_api_max_tokens', fallback='4096')
        custom_openai2_api_timeout = config_parser_object.get('API', 'custom_openai2_api_timeout', fallback='90')
        custom_openai2_api_retries = config_parser_object.get('API', 'custom_openai2_api_retry', fallback='3')
        custom_openai2_api_retry_delay = config_parser_object.get('API', 'custom_openai2_api_retry_delay', fallback='5')

        # Logging Checks for Local API IP loads
        # logging.debug(f"Loaded Kobold API IP: {kobold_api_ip}")
        # logging.debug(f"Loaded Llama API IP: {llama_api_IP}")
        # logging.debug(f"Loaded Ooba API IP: {ooba_api_IP}")
        # logging.debug(f"Loaded Tabby API IP: {tabby_api_IP}")
        # logging.debug(f"Loaded VLLM API URL: {vllm_api_url}")

        # Retrieve default API choices from the configuration file
        default_api = config_parser_object.get('API', 'default_api', fallback='openai')

        # Retrieve LLM API settings from the configuration file
        local_api_retries = config_parser_object.get('Local-API', 'Settings', fallback='3')
        local_api_retry_delay = config_parser_object.get('Local-API', 'local_api_retry_delay', fallback='5')

        # Retrieve output paths from the configuration file
        output_path = config_parser_object.get('Paths', 'output_path', fallback='results')
        _buffered_log("TRACE", f"Output path set to: {output_path}")

        # Save video transcripts
        save_video_transcripts = config_parser_object.get('Paths', 'save_video_transcripts', fallback='True')

        # Retrieve logging settings from the configuration file
        config_parser_object.get('Logging', 'log_level', fallback='INFO')
        config_parser_object.get('Logging', 'log_file', fallback='./Logs/tldw_logs.json')
        config_parser_object.get('Logging', 'log_metrics_file', fallback='./Logs/tldw_metrics_logs.json')

        # Retrieve processing choice from the configuration file
        processing_choice = config_parser_object.get('Processing', 'processing_choice', fallback='cpu')
        _buffered_log("TRACE", f"Processing choice set to: {processing_choice}")

        # [Chunking]
        # # Chunking Defaults
        # #
        # # Default Chunking Options for each media type
        chunking_method = config_parser_object.get('Chunking', 'chunking_method', fallback='words')
        chunk_max_size = config_parser_object.get('Chunking', 'chunk_max_size', fallback='400')
        chunk_overlap = config_parser_object.get('Chunking', 'chunk_overlap', fallback='200')
        adaptive_chunking = config_parser_object.get('Chunking', 'adaptive_chunking', fallback='False')
        chunking_multi_level = config_parser_object.get('Chunking', 'chunking_multi_level', fallback='False')
        chunk_language = config_parser_object.get('Chunking', 'chunk_language', fallback='en')
        #
        # Article Chunking
        article_chunking_method = config_parser_object.get('Chunking', 'article_chunking_method', fallback='words')
        article_chunk_max_size = config_parser_object.get('Chunking', 'article_chunk_max_size', fallback='400')
        article_chunk_overlap = config_parser_object.get('Chunking', 'article_chunk_overlap', fallback='200')
        article_adaptive_chunking = config_parser_object.get('Chunking', 'article_adaptive_chunking', fallback='False')
        article_chunking_multi_level = config_parser_object.get('Chunking', 'article_chunking_multi_level', fallback='False')
        article_language = config_parser_object.get('Chunking', 'article_language', fallback='english')
        #
        # Audio file Chunking
        audio_chunking_method = config_parser_object.get('Chunking', 'audio_chunking_method', fallback='words')
        audio_chunk_max_size = config_parser_object.get('Chunking', 'audio_chunk_max_size', fallback='400')
        audio_chunk_overlap = config_parser_object.get('Chunking', 'audio_chunk_overlap', fallback='200')
        audio_adaptive_chunking = config_parser_object.get('Chunking', 'audio_adaptive_chunking', fallback='False')
        audio_chunking_multi_level = config_parser_object.get('Chunking', 'audio_chunking_multi_level', fallback='False')
        audio_language = config_parser_object.get('Chunking', 'audio_language', fallback='english')
        #
        # Book Chunking
        book_chunking_method = config_parser_object.get('Chunking', 'book_chunking_method', fallback='words')
        book_chunk_max_size = config_parser_object.get('Chunking', 'book_chunk_max_size', fallback='400')
        book_chunk_overlap = config_parser_object.get('Chunking', 'book_chunk_overlap', fallback='200')
        book_adaptive_chunking = config_parser_object.get('Chunking', 'book_adaptive_chunking', fallback='False')
        book_chunking_multi_level = config_parser_object.get('Chunking', 'book_chunking_multi_level', fallback='False')
        book_language = config_parser_object.get('Chunking', 'book_language', fallback='english')
        #
        # Document Chunking
        document_chunking_method = config_parser_object.get('Chunking', 'document_chunking_method', fallback='words')
        document_chunk_max_size = config_parser_object.get('Chunking', 'document_chunk_max_size', fallback='400')
        document_chunk_overlap = config_parser_object.get('Chunking', 'document_chunk_overlap', fallback='200')
        document_adaptive_chunking = config_parser_object.get('Chunking', 'document_adaptive_chunking', fallback='False')
        document_chunking_multi_level = config_parser_object.get('Chunking', 'document_chunking_multi_level', fallback='False')
        document_language = config_parser_object.get('Chunking', 'document_language', fallback='english')
        #
        # Mediawiki Article Chunking
        mediawiki_article_chunking_method = config_parser_object.get('Chunking', 'mediawiki_article_chunking_method', fallback='words')
        mediawiki_article_chunk_max_size = config_parser_object.get('Chunking', 'mediawiki_article_chunk_max_size', fallback='400')
        mediawiki_article_chunk_overlap = config_parser_object.get('Chunking', 'mediawiki_article_chunk_overlap', fallback='200')
        mediawiki_article_adaptive_chunking = config_parser_object.get('Chunking', 'mediawiki_article_adaptive_chunking', fallback='False')
        mediawiki_article_chunking_multi_level = config_parser_object.get('Chunking', 'mediawiki_article_chunking_multi_level', fallback='False')
        mediawiki_article_language = config_parser_object.get('Chunking', 'mediawiki_article_language', fallback='english')
        #
        # Mediawiki Dump Chunking
        mediawiki_dump_chunking_method = config_parser_object.get('Chunking', 'mediawiki_dump_chunking_method', fallback='words')
        mediawiki_dump_chunk_max_size = config_parser_object.get('Chunking', 'mediawiki_dump_chunk_max_size', fallback='400')
        mediawiki_dump_chunk_overlap = config_parser_object.get('Chunking', 'mediawiki_dump_chunk_overlap', fallback='200')
        mediawiki_dump_adaptive_chunking = config_parser_object.get('Chunking', 'mediawiki_dump_adaptive_chunking', fallback='False')
        mediawiki_dump_chunking_multi_level = config_parser_object.get('Chunking', 'mediawiki_dump_chunking_multi_level', fallback='False')
        mediawiki_dump_language = config_parser_object.get('Chunking', 'mediawiki_dump_language', fallback='english')
        #
        # Obsidian Note Chunking
        obsidian_note_chunking_method = config_parser_object.get('Chunking', 'obsidian_note_chunking_method', fallback='words')
        obsidian_note_chunk_max_size = config_parser_object.get('Chunking', 'obsidian_note_chunk_max_size', fallback='400')
        obsidian_note_chunk_overlap = config_parser_object.get('Chunking', 'obsidian_note_chunk_overlap', fallback='200')
        obsidian_note_adaptive_chunking = config_parser_object.get('Chunking', 'obsidian_note_adaptive_chunking', fallback='False')
        obsidian_note_chunking_multi_level = config_parser_object.get('Chunking', 'obsidian_note_chunking_multi_level', fallback='False')
        obsidian_note_language = config_parser_object.get('Chunking', 'obsidian_note_language', fallback='english')
        #
        # Podcast Chunking
        podcast_chunking_method = config_parser_object.get('Chunking', 'podcast_chunking_method', fallback='words')
        podcast_chunk_max_size = config_parser_object.get('Chunking', 'podcast_chunk_max_size', fallback='400')
        podcast_chunk_overlap = config_parser_object.get('Chunking', 'podcast_chunk_overlap', fallback='200')
        podcast_adaptive_chunking = config_parser_object.get('Chunking', 'podcast_adaptive_chunking', fallback='False')
        podcast_chunking_multi_level = config_parser_object.get('Chunking', 'podcast_chunking_multi_level', fallback='False')
        podcast_language = config_parser_object.get('Chunking', 'podcast_language', fallback='english')
        #
        # Text Chunking
        text_chunking_method = config_parser_object.get('Chunking', 'text_chunking_method', fallback='words')
        text_chunk_max_size = config_parser_object.get('Chunking', 'text_chunk_max_size', fallback='400')
        text_chunk_overlap = config_parser_object.get('Chunking', 'text_chunk_overlap', fallback='200')
        text_adaptive_chunking = config_parser_object.get('Chunking', 'text_adaptive_chunking', fallback='False')
        text_chunking_multi_level = config_parser_object.get('Chunking', 'text_chunking_multi_level', fallback='False')
        text_language = config_parser_object.get('Chunking', 'text_language', fallback='english')
        #
        # Video Transcription Chunking
        video_chunking_method = config_parser_object.get('Chunking', 'video_chunking_method', fallback='words')
        video_chunk_max_size = config_parser_object.get('Chunking', 'video_chunk_max_size', fallback='400')
        video_chunk_overlap = config_parser_object.get('Chunking', 'video_chunk_overlap', fallback='200')
        video_adaptive_chunking = config_parser_object.get('Chunking', 'video_adaptive_chunking', fallback='False')
        video_chunking_multi_level = config_parser_object.get('Chunking', 'video_chunking_multi_level', fallback='False')
        video_language = config_parser_object.get('Chunking', 'video_language', fallback='english')
        #
        # Proposition Chunking Defaults
        proposition_engine = config_parser_object.get('Chunking', 'proposition_engine', fallback='heuristic')
        proposition_prompt_profile = config_parser_object.get('Chunking', 'proposition_prompt_profile', fallback='generic')
        proposition_aggressiveness = config_parser_object.get('Chunking', 'proposition_aggressiveness', fallback='1')
        proposition_min_proposition_length = config_parser_object.get('Chunking', 'proposition_min_proposition_length', fallback='15')
        #

        # Retrieve Embedding model settings from the configuration file
        # Default to Qwen3-Embedding-4B-GGUF if not specified
        embedding_model = config_parser_object.get('Embeddings', 'embedding_model', fallback='Qwen/Qwen3-Embedding-4B-GGUF')
        _buffered_log("TRACE", f"Embedding model set to: {embedding_model}")
        embedding_provider = config_parser_object.get('Embeddings', 'embedding_provider', fallback='huggingface')
        # Note: duplicate line removed - embedding_model already retrieved above
        onnx_model_path = config_parser_object.get('Embeddings', 'onnx_model_path', fallback="./App_Function_Libraries/onnx_models/text-embedding-3-small.onnx")
        model_dir = config_parser_object.get('Embeddings', 'model_dir', fallback="./App_Function_Libraries/onnx_models")
        embedding_api_url = config_parser_object.get('Embeddings', 'embedding_api_url', fallback="http://localhost:8080/v1/embeddings")
        embedding_api_key = config_parser_object.get('Embeddings', 'embedding_api_key', fallback='')
        # Fallback model if primary model fails
        embedding_fallback_model = config_parser_object.get('Embeddings', 'embedding_fallback_model', fallback='sentence-transformers/all-MiniLM-L6-v2')
        # Auto-generate embeddings on upload
        auto_generate_embeddings = config_parser_object.get('Embeddings', 'auto_generate_on_upload', fallback='false').lower() == 'true'
        chunk_size = config_parser_object.get('Embeddings', 'chunk_size', fallback=400)
        overlap = config_parser_object.get('Embeddings', 'overlap', fallback=200)
        # Contextual chunking defaults for embeddings
        enable_contextual_chunking_cfg = config_parser_object.get('Embeddings', 'enable_contextual_chunking', fallback='false')
        try:
            enable_contextual_chunking_flag = str(enable_contextual_chunking_cfg).strip().lower() in {"true","1","yes","on"}
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            enable_contextual_chunking_flag = False
        # Allow contextual LLM model in Embeddings (fallback to Claims section for backward compat)
        contextual_llm_model_cfg = config_parser_object.get('Embeddings', 'contextual_llm_model', fallback=None)
        contextual_llm_provider_cfg = config_parser_object.get('Embeddings', 'contextual_llm_provider', fallback=None)
        # Temperature for contextualization LLM
        contextual_llm_temperature_cfg = None
        try:
            _temp_val = config_parser_object.get('Embeddings', 'contextual_llm_temperature', fallback='')
            if _temp_val is not None and str(_temp_val).strip() != '':
                contextual_llm_temperature_cfg = float(_temp_val)
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            contextual_llm_temperature_cfg = None
        if not contextual_llm_model_cfg:
            contextual_llm_model_cfg = config_parser_object.get('Claims', 'contextual_llm_model', fallback=None)
        # Window size: allow None to lock full-doc behavior
        context_window_size_cfg = config_parser_object.get('Embeddings', 'context_window_size', fallback=None)
        if context_window_size_cfg is None:
            # Fallback to chunking section if not specified under Embeddings
            context_window_size_cfg = config_parser_object.get('Chunking', 'context_window_size', fallback=None)
        def _parse_optional_int(val):
            if val is None:
                return None
            s = str(val).strip().lower()
            if s in {"", "none", "null"}:
                return None
            try:
                return int(s)
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                return None
        context_window_size_val = _parse_optional_int(context_window_size_cfg)
        # Strategy/budget
        context_strategy_cfg = config_parser_object.get('Embeddings', 'context_strategy', fallback='auto')
        context_strategy_val = str(context_strategy_cfg).strip().lower() if context_strategy_cfg else 'auto'
        context_token_budget_cfg = config_parser_object.get('Embeddings', 'context_token_budget', fallback='6000')
        try:
            context_token_budget_val = int(str(context_token_budget_cfg).strip())
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            context_token_budget_val = 6000

        # Prompts - FIXME
        config_parser_object.get('Prompts', 'prompt_path', fallback='Databases/prompts.db')

        # Chat Dictionaries
        enable_chat_dictionaries = config_parser_object.get('Chat-Dictionaries', 'enable_chat_dictionaries', fallback='False')
        post_gen_replacement = config_parser_object.get('Chat-Dictionaries', 'post_gen_replacement', fallback='False')
        post_gen_replacement_dict = config_parser_object.get('Chat-Dictionaries', 'post_gen_replacement_dict', fallback='')
        chat_dict_chat_prompts = config_parser_object.get('Chat-Dictionaries', 'chat_dictionary_chat_prompts', fallback='')
        chat_dict_rag_prompts = config_parser_object.get('Chat-Dictionaries', 'chat_dictionary_RAG_prompts', fallback='')
        chat_dict_replacement_strategy = config_parser_object.get('Chat-Dictionaries', 'chat_dictionary_replacement_strategy', fallback='character_lore_first')
        chat_dict_max_tokens = config_parser_object.get('Chat-Dictionaries', 'chat_dictionary_max_tokens', fallback='1000')
        default_rag_prompt = config_parser_object.get('Chat-Dictionaries', 'default_rag_prompt', fallback='')
        rag_default_llm_provider = os.getenv("RAG_DEFAULT_LLM_PROVIDER") or (
            config_parser_object.get('RAG', 'default_llm_provider', fallback=None)
            if config_parser_object.has_section('RAG') else None
        )
        rag_default_llm_model = os.getenv("RAG_DEFAULT_LLM_MODEL") or (
            config_parser_object.get('RAG', 'default_llm_model', fallback=None)
            if config_parser_object.has_section('RAG') else None
        )

        # Auto-Save Values
        save_character_chats = config_parser_object.get('Auto-Save', 'save_character_chats', fallback='False')
        save_rag_chats = config_parser_object.get('Auto-Save', 'save_rag_chats', fallback='False')

        # Media Processing Limits
        max_audio_file_size_mb = int(config_parser_object.get('Media-Processing', 'max_audio_file_size_mb', fallback='500'))
        max_pdf_file_size_mb = int(config_parser_object.get('Media-Processing', 'max_pdf_file_size_mb', fallback='50'))
        max_video_file_size_mb = int(config_parser_object.get('Media-Processing', 'max_video_file_size_mb', fallback='1000'))
        max_epub_file_size_mb = int(config_parser_object.get('Media-Processing', 'max_epub_file_size_mb', fallback='100'))
        max_document_file_size_mb = int(config_parser_object.get('Media-Processing', 'max_document_file_size_mb', fallback='50'))
        max_code_file_size_mb = int(config_parser_object.get('Media-Processing', 'max_code_file_size_mb', fallback=str(max_document_file_size_mb)))
        # Processing timeouts
        pdf_conversion_timeout_seconds = int(config_parser_object.get('Media-Processing', 'pdf_conversion_timeout_seconds', fallback='300'))
        audio_processing_timeout_seconds = int(config_parser_object.get('Media-Processing', 'audio_processing_timeout_seconds', fallback='600'))
        video_processing_timeout_seconds = int(config_parser_object.get('Media-Processing', 'video_processing_timeout_seconds', fallback='1200'))
        # Archive processing limits
        max_archive_internal_files = int(config_parser_object.get('Media-Processing', 'max_archive_internal_files', fallback='100'))
        max_archive_uncompressed_size_mb = int(config_parser_object.get('Media-Processing', 'max_archive_uncompressed_size_mb', fallback='200'))
        max_archive_nesting_depth = int(config_parser_object.get('Media-Processing', 'max_archive_nesting_depth', fallback='2'))
        max_archive_member_uncompressed_size_mb = int(
            config_parser_object.get(
                'Media-Processing',
                'max_archive_member_uncompressed_size_mb',
                fallback='100',
            )
        )
        sanitize_html_uploads = is_truthy(
            config_parser_object.get(
                'Media-Processing',
                'sanitize_html_uploads',
                fallback='true',
            )
        )
        sanitize_xml_uploads = is_truthy(
            config_parser_object.get(
                'Media-Processing',
                'sanitize_xml_uploads',
                fallback='true',
            )
        )
        sanitize_email_html_bodies = is_truthy(
            config_parser_object.get(
                'Media-Processing',
                'sanitize_email_html_bodies',
                fallback='true',
            )
        )
        validate_email_archive_contents = is_truthy(
            config_parser_object.get(
                'Media-Processing',
                'validate_email_archive_contents',
                fallback='true',
            )
        )
        # Transcription settings
        audio_transcription_buffer_size_mb = int(config_parser_object.get('Media-Processing', 'audio_transcription_buffer_size_mb', fallback='10'))
        # General settings
        uuid_generation_length = int(config_parser_object.get('Media-Processing', 'uuid_generation_length', fallback='8'))
        # Kept video storage limits
        kept_video_max_files = int(config_parser_object.get('Media-Processing', 'kept_video_max_files', fallback='5'))
        kept_video_max_storage_mb = int(config_parser_object.get('Media-Processing', 'kept_video_max_storage_mb', fallback='500'))
        kept_video_retention_hours = int(config_parser_object.get('Media-Processing', 'kept_video_retention_hours', fallback='2'))

        # Local API Timeout
        local_api_timeout = config_parser_object.get('Local-API', 'local_api_timeout', fallback='90')

        # STT Settings
        raw_default_stt_provider = config_parser_object.get('STT-Settings', 'default_stt_provider', fallback='faster-whisper')
        raw_default_transcriber = config_parser_object.get('STT-Settings', 'default_transcriber', fallback='')

        def _normalize_stt_provider_name(name: str) -> str:
            """
            Normalize STT provider/transcriber identifiers from config.

            - Accepts both 'faster_whisper' and 'faster-whisper' and normalizes to 'faster-whisper'.
            - Returns lower-cased identifiers for consistency.
            """
            name = (name or "").strip()
            if not name:
                return ""
            lowered = name.lower()
            if lowered in ("faster-whisper", "faster_whisper"):
                return "faster-whisper"
            if lowered in ("vibevoice", "vibevoice-asr", "vibevoice_asr"):
                return "vibevoice"
            return lowered

        default_stt_provider = _normalize_stt_provider_name(raw_default_stt_provider)
        tmp_default_transcriber = _normalize_stt_provider_name(raw_default_transcriber)
        # If default_transcriber is not set explicitly, fall back to default_stt_provider
        default_transcriber = tmp_default_transcriber or default_stt_provider
        default_batch_transcription_model = (
            config_parser_object.get(
                'STT-Settings',
                'default_batch_transcription_model',
                fallback='parakeet-tdt-0.6b-v3-onnx',
            ).strip()
            or 'parakeet-tdt-0.6b-v3-onnx'
        )
        default_streaming_transcription_model = (
            config_parser_object.get(
                'STT-Settings',
                'default_streaming_transcription_model',
                fallback='parakeet-tdt-0.6b-v3-onnx',
            ).strip()
            or 'parakeet-tdt-0.6b-v3-onnx'
        )
        nemo_model_variant = config_parser_object.get('STT-Settings', 'nemo_model_variant', fallback='onnx')
        nemo_device = config_parser_object.get('STT-Settings', 'nemo_device', fallback='cuda')
        nemo_cache_dir = config_parser_object.get('STT-Settings', 'nemo_cache_dir', fallback='./models/nemo')
        # STT custom vocabulary (optional)
        stt_custom_vocab_terms_file = config_parser_object.get('STT-Settings', 'custom_vocab_terms_file', fallback='')
        stt_custom_vocab_replacements_file = config_parser_object.get('STT-Settings', 'custom_vocab_replacements_file', fallback='')
        stt_custom_vocab_initial_prompt_enable = config_parser_object.get('STT-Settings', 'custom_vocab_initial_prompt_enable', fallback='True')
        stt_custom_vocab_postprocess_enable = config_parser_object.get('STT-Settings', 'custom_vocab_postprocess_enable', fallback='True')
        stt_custom_vocab_prompt_template = config_parser_object.get('STT-Settings', 'custom_vocab_prompt_template', fallback='')
        stt_custom_vocab_case_sensitive = config_parser_object.get('STT-Settings', 'custom_vocab_case_sensitive', fallback='False')

        # Diarization Settings (optional; overrides DIARIZATION_CONFIG when present)
        def _get_bool(section: str, key: str, default: bool) -> bool:
            try:
                raw = config_parser_object.get(section, key, fallback=None)
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                return default
            if raw is None:
                return default
            s = str(raw).strip().lower()
            if is_truthy(s):
                return True
            if s in {"0", "false", "no", "n", "off"}:
                return False
            return default

        def _get_float(section: str, key: str, default: float) -> float:
            try:
                raw = config_parser_object.get(section, key, fallback=None)
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                return default
            if raw is None or str(raw).strip() == "":
                return default
            try:
                return float(str(raw).strip())
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                return default

        def _get_int(section: str, key: str, default: int) -> int:
            try:
                raw = config_parser_object.get(section, key, fallback=None)
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                return default
            if raw is None or str(raw).strip() == "":
                return default
            try:
                return int(str(raw).strip())
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                return default

        def _get_str(section: str, key: str, default: str | None) -> str | None:
            try:
                raw = config_parser_object.get(section, key, fallback=None)
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                return default
            if raw is None:
                return default
            s = str(raw).strip()
            return s if s != "" else default

        def _to_optional_int(raw: str | None) -> int | None:
            """Parse an optional integer config value, returning ``None`` when absent/invalid."""
            if raw is None:
                return None
            try:
                return int(str(raw).strip())
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                return None

        def _to_optional_float(raw: str | None) -> float | None:
            """Parse an optional float config value, returning ``None`` when absent/invalid."""
            if raw is None:
                return None
            try:
                return float(str(raw).strip())
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                return None

        # Parakeet MLX settings
        mlx_model_id = _get_str('STT-Settings', 'mlx_model_id', 'mlx-community/parakeet-tdt-0.6b-v3')
        mlx_cache_dir = _get_str('STT-Settings', 'mlx_cache_dir', '') or ''
        parakeet_onnx_model_id = (
            _get_str('STT-Settings', 'parakeet_onnx_model_id', 'istupakov/parakeet-tdt-0.6b-v3-onnx')
            or 'istupakov/parakeet-tdt-0.6b-v3-onnx'
        )
        parakeet_onnx_revision = _get_str('STT-Settings', 'parakeet_onnx_revision', None)
        mlx_chunk_duration = _get_float('STT-Settings', 'mlx_chunk_duration', 30.0)
        mlx_overlap_duration = _get_float('STT-Settings', 'mlx_overlap_duration', 5.0)
        buffered_chunk_duration = _get_float('STT-Settings', 'buffered_chunk_duration', mlx_chunk_duration)
        buffered_total_buffer = _to_optional_float(_get_str('STT-Settings', 'buffered_total_buffer', None))
        buffered_merge_algo = _get_str('STT-Settings', 'buffered_merge_algo', 'middle') or 'middle'
        streaming_fallback_to_whisper = _get_bool(
            'STT-Settings',
            'streaming_fallback_to_whisper',
            default=False,
        )

        mlx_decoding_mode = _get_str('STT-Settings', 'mlx_decoding_mode', '') or ''
        mlx_beam_size = _to_optional_int(_get_str('STT-Settings', 'mlx_beam_size', None))
        mlx_length_penalty = _to_optional_float(_get_str('STT-Settings', 'mlx_length_penalty', None))
        mlx_patience = _to_optional_float(_get_str('STT-Settings', 'mlx_patience', None))
        mlx_duration_reward = _to_optional_float(_get_str('STT-Settings', 'mlx_duration_reward', None))
        mlx_sentence_max_words = _to_optional_int(_get_str('STT-Settings', 'mlx_sentence_max_words', None))
        mlx_sentence_silence_gap = _to_optional_float(_get_str('STT-Settings', 'mlx_sentence_silence_gap', None))
        mlx_sentence_max_duration = _to_optional_float(_get_str('STT-Settings', 'mlx_sentence_max_duration', None))
        mlx_stream_context_left = _get_int('STT-Settings', 'mlx_stream_context_left', 256)
        mlx_stream_context_right = _get_int('STT-Settings', 'mlx_stream_context_right', 256)
        mlx_stream_depth = _get_int('STT-Settings', 'mlx_stream_depth', 1)
        mlx_stream_keep_original_attention = _get_bool(
            'STT-Settings',
            'mlx_stream_keep_original_attention',
            default=False,
        )

        def _env_or_cfg_bool(env_key: str, section: str, key: str, default: bool) -> bool:
            env_val = os.getenv(env_key)
            if env_val is not None:
                return is_truthy(env_val)
            return _get_bool(section, key, default)

        def _env_or_cfg_int(env_key: str, section: str, key: str, default: int) -> int:
            env_val = os.getenv(env_key)
            if env_val is not None:
                try:
                    return int(str(env_val).strip())
                except _CONFIG_NONCRITICAL_EXCEPTIONS:
                    return default
            return _get_int(section, key, default)

        def _env_or_cfg_str(env_key: str, section: str, key: str, default: str | None) -> str | None:
            env_val = os.getenv(env_key)
            if env_val is not None:
                s = str(env_val).strip()
                return s if s != "" else default
            return _get_str(section, key, default)

        # VibeVoice-ASR settings (local inference + optional vLLM HTTP path)
        vibevoice_enabled = _get_bool('STT-Settings', 'vibevoice_enabled', default=False)
        vibevoice_model_id = _get_str('STT-Settings', 'vibevoice_model_id', 'microsoft/VibeVoice-ASR') or 'microsoft/VibeVoice-ASR'
        vibevoice_device = _get_str('STT-Settings', 'vibevoice_device', 'cuda') or 'cuda'
        vibevoice_dtype = _get_str('STT-Settings', 'vibevoice_dtype', 'bfloat16') or 'bfloat16'
        vibevoice_cache_dir = _get_str('STT-Settings', 'vibevoice_cache_dir', './models/vibevoice') or './models/vibevoice'
        vibevoice_allow_download = _get_bool('STT-Settings', 'vibevoice_allow_download', default=True)
        vibevoice_vllm_enabled = _get_bool('STT-Settings', 'vibevoice_vllm_enabled', default=False)
        vibevoice_vllm_base_url = _get_str('STT-Settings', 'vibevoice_vllm_base_url', '') or ''
        vibevoice_vllm_model_id = _get_str('STT-Settings', 'vibevoice_vllm_model_id', vibevoice_model_id) or vibevoice_model_id
        vibevoice_vllm_api_key = _get_str('STT-Settings', 'vibevoice_vllm_api_key', None)
        vibevoice_vllm_timeout_seconds = _get_int('STT-Settings', 'vibevoice_vllm_timeout_seconds', 600)

        diarization_config = dict(DIARIZATION_CONFIG)
        if config_parser_object.has_section('Diarization'):
            # Backend and VAD behavior
            diarization_config['backend'] = _get_str('Diarization', 'backend', diarization_config.get('backend')) or diarization_config.get('backend')
            diarization_config['vad_backend'] = _get_str('Diarization', 'vad_backend', diarization_config.get('vad_backend')) or diarization_config.get('vad_backend')
            diarization_config['vad_threshold'] = _get_float('Diarization', 'vad_threshold', diarization_config.get('vad_threshold', 0.5))
            diarization_config['vad_min_speech_duration'] = _get_float('Diarization', 'vad_min_speech_duration', diarization_config.get('vad_min_speech_duration', 0.25))
            diarization_config['vad_min_silence_duration'] = _get_float('Diarization', 'vad_min_silence_duration', diarization_config.get('vad_min_silence_duration', 0.25))
            diarization_config['allow_vad_fallback'] = _get_bool('Diarization', 'allow_vad_fallback', diarization_config.get('allow_vad_fallback', True))
            diarization_config['enable_torch_hub_fetch'] = _get_bool('Diarization', 'enable_torch_hub_fetch', diarization_config.get('enable_torch_hub_fetch', True))
            diarization_config['onnx_model_path'] = _get_str('Diarization', 'onnx_model_path', diarization_config.get('onnx_model_path'))

            # NeMo multitalk settings
            diarization_config['nemo_multitalk_asr_model'] = _get_str('Diarization', 'nemo_multitalk_asr_model', diarization_config.get('nemo_multitalk_asr_model')) or diarization_config.get('nemo_multitalk_asr_model')
            diarization_config['nemo_multitalk_diar_model'] = _get_str('Diarization', 'nemo_multitalk_diar_model', diarization_config.get('nemo_multitalk_diar_model')) or diarization_config.get('nemo_multitalk_diar_model')
            diarization_config['nemo_multitalk_device'] = _get_str('Diarization', 'nemo_multitalk_device', diarization_config.get('nemo_multitalk_device')) or diarization_config.get('nemo_multitalk_device')
            diarization_config['nemo_multitalk_cache_dir'] = _get_str('Diarization', 'nemo_multitalk_cache_dir', diarization_config.get('nemo_multitalk_cache_dir')) or diarization_config.get('nemo_multitalk_cache_dir')
            diarization_config['nemo_multitalk_max_speakers'] = _get_int('Diarization', 'nemo_multitalk_max_speakers', diarization_config.get('nemo_multitalk_max_speakers', 4))
            diarization_config['nemo_multitalk_disable_cuda_graphs'] = _get_bool('Diarization', 'nemo_multitalk_disable_cuda_graphs', diarization_config.get('nemo_multitalk_disable_cuda_graphs', True))

            # Segmentation
            diarization_config['segment_duration'] = _get_float('Diarization', 'segment_duration', diarization_config.get('segment_duration', 30.0))
            diarization_config['segment_overlap'] = _get_float('Diarization', 'segment_overlap', diarization_config.get('segment_overlap', 0.5))
            diarization_config['min_segment_duration'] = _get_float('Diarization', 'min_segment_duration', diarization_config.get('min_segment_duration', 1.0))
            diarization_config['max_segment_duration'] = _get_float('Diarization', 'max_segment_duration', diarization_config.get('max_segment_duration', 3.0))

            # Embeddings & clustering
            diarization_config['embedding_model'] = _get_str('Diarization', 'embedding_model', diarization_config.get('embedding_model')) or diarization_config.get('embedding_model')
            diarization_config['embedding_device'] = _get_str('Diarization', 'embedding_device', diarization_config.get('embedding_device')) or diarization_config.get('embedding_device')
            diarization_config['embedding_local_only'] = _get_bool('Diarization', 'embedding_local_only', diarization_config.get('embedding_local_only', False))
            diarization_config['embedding_batch_size'] = _get_int('Diarization', 'embedding_batch_size', diarization_config.get('embedding_batch_size', 32))
            diarization_config['clustering_method'] = _get_str('Diarization', 'clustering_method', diarization_config.get('clustering_method')) or diarization_config.get('clustering_method')
            diarization_config['similarity_threshold'] = _get_float('Diarization', 'similarity_threshold', diarization_config.get('similarity_threshold', 0.85))
            diarization_config['min_speakers'] = _get_int('Diarization', 'min_speakers', diarization_config.get('min_speakers', 1))
            diarization_config['max_speakers'] = _get_int('Diarization', 'max_speakers', diarization_config.get('max_speakers', 10))

            # Post-processing and memory
            diarization_config['merge_threshold'] = _get_float('Diarization', 'merge_threshold', diarization_config.get('merge_threshold', 0.5))
            diarization_config['min_speaker_duration'] = _get_float('Diarization', 'min_speaker_duration', diarization_config.get('min_speaker_duration', 3.0))
            diarization_config['detect_overlapping_speech'] = _get_bool('Diarization', 'detect_overlapping_speech', diarization_config.get('detect_overlapping_speech', False))
            diarization_config['overlap_confidence_threshold'] = _get_float('Diarization', 'overlap_confidence_threshold', diarization_config.get('overlap_confidence_threshold', 0.7))
            diarization_config['memory_efficient'] = _get_bool('Diarization', 'memory_efficient', diarization_config.get('memory_efficient', False))
            diarization_config['max_memory_mb'] = _get_int('Diarization', 'max_memory_mb', diarization_config.get('max_memory_mb', 2048))

        # TTS Settings
        _tts_placeholder_literals = {
            "",
            "FIXME",
            "TODO",
            "TBD",
            "CHANGE_ME",
            "CHANGE-ME",
            "PLACEHOLDER",
            "NONE",
            "NULL",
            "N/A",
            "NA",
        }

        def _get_tts_setting(option: str, fallback: str) -> str:
            raw = config_parser_object.get('TTS-Settings', option, fallback=fallback)
            value = str(raw).strip()
            return fallback if value.upper() in _tts_placeholder_literals else value

        local_tts_device = _get_tts_setting('local_tts_device', 'cpu')
        default_tts_provider = _get_tts_setting('default_tts_provider', 'openai')
        tts_voice = _get_tts_setting('default_tts_voice', 'shimmer')
        tts_history_enabled = _env_or_cfg_bool("TTS_HISTORY_ENABLED", "TTS-Settings", "tts_history_enabled", True)
        tts_history_store_text = _env_or_cfg_bool("TTS_HISTORY_STORE_TEXT", "TTS-Settings", "tts_history_store_text", True)
        tts_history_store_failed = _env_or_cfg_bool("TTS_HISTORY_STORE_FAILED", "TTS-Settings", "tts_history_store_failed", True)
        tts_history_hash_key = _env_or_cfg_str("TTS_HISTORY_HASH_KEY", "TTS-Settings", "tts_history_hash_key", None)
        tts_history_retention_days = _env_or_cfg_int("TTS_HISTORY_RETENTION_DAYS", "TTS-Settings", "tts_history_retention_days", 90)
        tts_history_max_rows_per_user = _env_or_cfg_int("TTS_HISTORY_MAX_ROWS_PER_USER", "TTS-Settings", "tts_history_max_rows_per_user", 10000)
        tts_history_purge_interval_hours = _env_or_cfg_int(
            "TTS_HISTORY_PURGE_INTERVAL_HOURS",
            "TTS-Settings",
            "tts_history_purge_interval_hours",
            24,
        )
        # Open AI TTS
        default_openai_tts_model = _get_tts_setting('default_openai_tts_model', 'tts-1-hd')
        default_openai_tts_voice = _get_tts_setting('default_openai_tts_voice', 'shimmer')
        default_openai_tts_speed = _get_tts_setting('default_openai_tts_speed', '1')
        default_openai_tts_output_format = _get_tts_setting('default_openai_tts_output_format', 'mp3')
        config_parser_object.get('TTS-Settings', 'default_openai_tts_streaming', fallback='False')
        # Google TTS
        default_google_tts_model = _get_tts_setting('default_google_tts_model', 'en-US')
        default_google_tts_voice = _get_tts_setting('default_google_tts_voice', 'en-US-Neural2-A')
        default_google_tts_speed = _get_tts_setting('default_google_tts_speed', '1')
        # ElevenLabs TTS
        default_eleven_tts_model = _get_tts_setting('default_eleven_tts_model', 'eleven_monolingual_v1')
        default_eleven_tts_voice = _get_tts_setting('default_eleven_tts_voice', 'pNInz6obpgDQGcFmaJgB')
        default_eleven_tts_language_code = _get_tts_setting('default_eleven_tts_language_code', 'en')
        default_eleven_tts_voice_stability = _get_tts_setting('default_eleven_tts_voice_stability', '0.5')
        default_eleven_tts_voice_similiarity_boost = _get_tts_setting('default_eleven_tts_voice_similiarity_boost', '0.75')
        default_eleven_tts_voice_style = _get_tts_setting('default_eleven_tts_voice_style', '0.0')
        default_eleven_tts_voice_use_speaker_boost = _get_tts_setting('default_eleven_tts_voice_use_speaker_boost', 'true')
        default_eleven_tts_output_format = _get_tts_setting('default_eleven_tts_output_format', 'mp3_44100_128')
        # AllTalk TTS
        alltalk_api_ip = config_parser_object.get('TTS-Settings', 'alltalk_api_ip', fallback='http://127.0.0.1:7851/v1/audio/speech')
        default_alltalk_tts_model = config_parser_object.get('TTS-Settings', 'default_alltalk_tts_model', fallback='alltalk_model')
        default_alltalk_tts_voice = config_parser_object.get('TTS-Settings', 'default_alltalk_tts_voice', fallback='alloy')
        default_alltalk_tts_speed = config_parser_object.get('TTS-Settings', 'default_alltalk_tts_speed', fallback=1.0)
        default_alltalk_tts_output_format = config_parser_object.get('TTS-Settings', 'default_alltalk_tts_output_format', fallback='mp3')

        # Kokoro TTS
        config_parser_object.get('TTS-Settings', 'kokoro_model_path', fallback='Databases/kokoro_models')
        default_kokoro_tts_model = config_parser_object.get('TTS-Settings', 'default_kokoro_tts_model', fallback='pht')
        default_kokoro_tts_voice = config_parser_object.get('TTS-Settings', 'default_kokoro_tts_voice', fallback='sky')
        default_kokoro_tts_speed = config_parser_object.get('TTS-Settings', 'default_kokoro_tts_speed', fallback=1.0)
        default_kokoro_tts_output_format = config_parser_object.get('TTS-Settings', 'default_kokoro_tts_output_format', fallback='wav')


        # Self-hosted OpenAI API TTS
        default_openai_api_tts_model = config_parser_object.get('TTS-Settings', 'default_openai_api_tts_model', fallback='tts-1-hd')
        default_openai_api_tts_voice = config_parser_object.get('TTS-Settings', 'default_openai_api_tts_voice', fallback='shimmer')
        default_openai_api_tts_speed = config_parser_object.get('TTS-Settings', 'default_openai_api_tts_speed', fallback='1')
        default_openai_api_tts_output_format = config_parser_object.get('TTS-Settings', 'default_openai_tts_api_output_format', fallback='mp3')
        default_openai_api_tts_streaming = config_parser_object.get('TTS-Settings', 'default_openai_tts_streaming', fallback='False')


        # Search Engines
        search_provider_default = config_parser_object.get('Search-Engines', 'search_provider_default', fallback='google')
        search_language_query = config_parser_object.get('Search-Engines', 'search_language_query', fallback='en')
        search_language_results = config_parser_object.get('Search-Engines', 'search_language_results', fallback='en')
        search_language_analysis = config_parser_object.get('Search-Engines', 'search_language_analysis', fallback='en')
        search_default_max_queries = 10
        search_enable_subquery = config_parser_object.get('Search-Engines', 'search_enable_subquery', fallback='True')
        search_enable_subquery_count_max = config_parser_object.get('Search-Engines', 'search_enable_subquery_count_max', fallback=5)
        search_result_rerank = config_parser_object.get('Search-Engines', 'search_result_rerank', fallback='True')
        search_result_max = config_parser_object.get('Search-Engines', 'search_result_max', fallback=10)
        search_result_max_per_query = config_parser_object.get('Search-Engines', 'search_result_max_per_query', fallback=10)
        search_result_blacklist = config_parser_object.get('Search-Engines', 'search_result_blacklist', fallback='')
        search_result_display_type = config_parser_object.get('Search-Engines', 'search_result_display_type', fallback='list')
        search_result_display_metadata = config_parser_object.get('Search-Engines', 'search_result_display_metadata', fallback='False')
        search_result_save_to_db = config_parser_object.get('Search-Engines', 'search_result_save_to_db', fallback='True')
        search_result_analysis_tone = config_parser_object.get('Search-Engines', 'search_result_analysis_tone', fallback='')
        relevance_analysis_llm = config_parser_object.get('Search-Engines', 'relevance_analysis_llm', fallback='False')
        final_answer_llm = config_parser_object.get('Search-Engines', 'final_answer_llm', fallback='False')
        # Search Engine Specifics
        baidu_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_baidu', fallback='')
        # Bing Search Settings
        bing_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_bing', fallback='')
        bing_country_code = config_parser_object.get('Search-Engines', 'search_engine_country_code_bing', fallback='us')
        bing_search_api_url = config_parser_object.get('Search-Engines', 'search_engine_api_url_bing', fallback='')
        # Brave Search Settings
        brave_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_brave_regular', fallback='')
        brave_search_ai_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_brave_ai', fallback='')
        brave_country_code = config_parser_object.get('Search-Engines', 'search_engine_country_code_brave', fallback='us')
        # DuckDuckGo Search Settings
        duckduckgo_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_duckduckgo', fallback='')
        # Google Search Settings
        google_search_api_url = config_parser_object.get('Search-Engines', 'search_engine_api_url_google', fallback='')
        google_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_google', fallback='')
        google_search_engine_id = config_parser_object.get('Search-Engines', 'search_engine_id_google', fallback='')
        google_simp_trad_chinese = config_parser_object.get('Search-Engines', 'enable_traditional_chinese', fallback='0')
        limit_google_search_to_country = config_parser_object.get('Search-Engines', 'limit_google_search_to_country', fallback='0')
        google_search_country = config_parser_object.get('Search-Engines', 'google_search_country', fallback='us')
        google_search_country_code = config_parser_object.get('Search-Engines', 'google_search_country_code', fallback='us')
        google_filter_setting = config_parser_object.get('Search-Engines', 'google_filter_setting', fallback='1')
        google_user_geolocation = config_parser_object.get('Search-Engines', 'google_user_geolocation', fallback='')
        google_ui_language = config_parser_object.get('Search-Engines', 'google_ui_language', fallback='en')
        google_limit_search_results_to_language = config_parser_object.get('Search-Engines', 'google_limit_search_results_to_language', fallback='')
        google_default_search_results = config_parser_object.get('Search-Engines', 'google_default_search_results', fallback='10')
        google_safe_search = config_parser_object.get('Search-Engines', 'google_safe_search', fallback='active')
        google_enable_site_search = config_parser_object.get('Search-Engines', 'google_enable_site_search', fallback='0')
        google_site_search_include = config_parser_object.get('Search-Engines', 'google_site_search_include', fallback='')
        google_site_search_exclude = config_parser_object.get('Search-Engines', 'google_site_search_exclude', fallback='')
        google_sort_results_by = config_parser_object.get('Search-Engines', 'google_sort_results_by', fallback='relevance')
        # Kagi Search Settings
        kagi_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_kagi', fallback='')
        # Searx Search Settings
        search_engine_searx_api = config_parser_object.get('Search-Engines', 'search_engine_searx_api', fallback='')
        # Tavily Search Settings
        tavily_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_tavily', fallback='')
        # Serper Search Settings
        serper_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_serper', fallback='')
        serper_search_api_url = config_parser_object.get('Search-Engines', 'search_engine_api_url_serper', fallback='https://google.serper.dev/search')
        # Exa Search Settings
        exa_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_exa', fallback='')
        exa_search_api_url = config_parser_object.get('Search-Engines', 'search_engine_api_url_exa', fallback='https://api.exa.ai/search')
        # Firecrawl Search Settings
        firecrawl_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_firecrawl', fallback='')
        firecrawl_search_api_url = config_parser_object.get('Search-Engines', 'search_engine_api_url_firecrawl', fallback='https://api.firecrawl.dev/v2/search')
        # Yandex Search Settings
        yandex_search_api_key = config_parser_object.get('Search-Engines', 'search_engine_api_key_yandex', fallback='')
        yandex_search_engine_id = config_parser_object.get('Search-Engines', 'search_engine_id_yandex', fallback='')

        # Prompts
        sub_question_generation_prompt = config_parser_object.get('Prompts', 'sub_question_generation_prompt', fallback='')
        search_result_relevance_eval_prompt = config_parser_object.get('Prompts', 'search_result_relevance_eval_prompt', fallback='')
        analyze_search_results_prompt = config_parser_object.get('Prompts', 'analyze_search_results_prompt', fallback='')

        # Web Scraper settings
        web_scraper_api_key = config_parser_object.get('Web-Scraper', 'web_scraper_api_key', fallback='')
        web_scraper_api_url = config_parser_object.get('Web-Scraper', 'web_scraper_api_url', fallback='')
        web_scraper_api_timeout = config_parser_object.get('Web-Scraper', 'web_scraper_api_timeout', fallback='90')
        web_scraper_api_retries = config_parser_object.get('Web-Scraper', 'web_scraper_api_retries', fallback='3')
        web_scraper_api_retry_delay = config_parser_object.get('Web-Scraper', 'web_scraper_api_retry_delay', fallback='5')
        web_scraper_retry_count = config_parser_object.get('Web-Scraper', 'web_scraper_retry_count', fallback='3')
        web_scraper_retry_timeout = config_parser_object.get('Web-Scraper', 'web_scraper_retry_timeout', fallback='5')
        web_scraper_stealth_playwright = config_parser_object.get('Web-Scraper', 'web_scraper_stealth_playwright', fallback='False')
        custom_scrapers_yaml_path = (
            os.getenv('WEB_SCRAPER_CUSTOM_SCRAPERS_YAML_PATH')
            or os.getenv('CUSTOM_SCRAPERS_YAML_PATH')
            or config_parser_object.get(
                'Web-Scraper',
                'custom_scrapers_yaml_path',
                fallback='tldw_Server_API/Config_Files/custom_scrapers.yaml',
            )
        )
        web_scraper_default_backend = (
            os.getenv('WEB_SCRAPER_HTTP_BACKEND')
            or os.getenv('WEB_SCRAPER_DEFAULT_BACKEND')
            or config_parser_object.get(
            'Web-Scraper',
            'web_scraper_default_backend',
            fallback='auto',
            )
        )
        web_scraper_ua_mode = os.getenv('WEB_SCRAPER_UA_MODE') or config_parser_object.get(
            'Web-Scraper',
            'web_scraper_ua_mode',
            fallback='fixed',
        )

        # Web Scraper crawl flags (env overrides config.txt)
        def _env_or_cfg(env_key: str, section: str, cfg_key: str, default: str) -> str:
            return os.getenv(env_key) or config_parser_object.get(section, cfg_key, fallback=default)

        def _as_bool(v: object, d: bool) -> bool:
            try:
                s = str(v).strip().lower()
                if is_truthy(s):
                    return True
                if s in {"0", "false", "no", "off", "n"}:
                    return False
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                pass
            return d

        def _as_int(v: object, d: int) -> int:
            try:
                return int(str(v))
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                return d

        def _as_float(v: object, d: float) -> float:
            try:
                return float(str(v))
            except _CONFIG_NONCRITICAL_EXCEPTIONS:
                return d

        web_crawl_strategy = _env_or_cfg('WEB_CRAWL_STRATEGY', 'Web-Scraper', 'web_crawl_strategy', 'default')
        web_crawl_include_external = _as_bool(
            _env_or_cfg('WEB_CRAWL_INCLUDE_EXTERNAL', 'Web-Scraper', 'web_crawl_include_external', 'false'), False
        )
        web_crawl_score_threshold = _as_float(
            _env_or_cfg('WEB_CRAWL_SCORE_THRESHOLD', 'Web-Scraper', 'web_crawl_score_threshold', '0.0'), 0.0
        )
        web_crawl_max_pages = _as_int(
            _env_or_cfg('WEB_CRAWL_MAX_PAGES', 'Web-Scraper', 'web_crawl_max_pages', '100'), 100
        )
        # Optional allow/block domain lists (CSV). Env overrides config.txt.
        web_crawl_allowed_domains = _env_or_cfg('WEB_CRAWL_ALLOWED_DOMAINS', 'Web-Scraper', 'web_crawl_allowed_domains', '')
        web_crawl_blocked_domains = _env_or_cfg('WEB_CRAWL_BLOCKED_DOMAINS', 'Web-Scraper', 'web_crawl_blocked_domains', '')
        # Optional robots-respect flag (bool). Env overrides config.txt.
        web_scraper_respect_robots = _as_bool(
            _env_or_cfg('WEB_SCRAPER_RESPECT_ROBOTS', 'Web-Scraper', 'web_scraper_respect_robots', 'true'), True
        )
        web_outbound_policy_mode_value = web_outbound_policy_mode()
        # Optional scorers configuration
        web_crawl_enable_keyword = _as_bool(
            _env_or_cfg('WEB_CRAWL_ENABLE_KEYWORD_SCORER', 'Web-Scraper', 'web_crawl_enable_keyword_scorer', 'false'), False
        )
        web_crawl_keywords = _env_or_cfg('WEB_CRAWL_KEYWORDS', 'Web-Scraper', 'web_crawl_keywords', '')
        web_crawl_enable_domain_map = _as_bool(
            _env_or_cfg('WEB_CRAWL_ENABLE_DOMAIN_MAP', 'Web-Scraper', 'web_crawl_enable_domain_map', 'false'), False
        )
        web_crawl_domain_map = _env_or_cfg('WEB_CRAWL_DOMAIN_MAP', 'Web-Scraper', 'web_crawl_domain_map', '')

        def _section_items_dict(section_name: str) -> dict[str, Any]:
            try:
                if hasattr(config_parser_object, "has_section") and config_parser_object.has_section(section_name):
                    return dict(config_parser_object.items(section_name))
            except _CONFIG_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Failed to read config section '{}': {}", section_name, exc)
            return {}

        from tldw_Server_API.app.core.config_sections.stt import load_stt_config

        stt_vnext_config = load_stt_config(config_parser_object)
        stt_vnext_items = dict(vars(stt_vnext_config))

        def _numbered_custom_openai_config(provider_number: int) -> Optional[dict[str, Any]]:
            api_ip = _env_or_config_option_value(
                custom_openai_endpoint_env_keys(provider_number),
                config_parser_object,
                'API',
                custom_openai_config_option_names(provider_number, 'ip'),
                fallback=None,
            )
            api_key = _env_or_config_option_value(
                custom_openai_api_key_env_keys(provider_number),
                config_parser_object,
                'API',
                custom_openai_config_option_names(provider_number, 'key'),
                fallback=None,
            )
            model = _env_or_config_option_value(
                custom_openai_model_env_keys(provider_number),
                config_parser_object,
                'API',
                custom_openai_config_option_names(provider_number, 'model'),
                fallback=None,
            )
            if not any(value for value in (api_ip, api_key, model)):
                return None

            return {
                'api_ip': api_ip,
                'api_key': api_key,
                'streaming': _first_config_option_value(
                    config_parser_object,
                    'API',
                    custom_openai_config_option_names(provider_number, 'streaming'),
                    fallback=custom_openai_api_streaming,
                ),
                'model': model,
                'temperature': _first_config_option_value(
                    config_parser_object,
                    'API',
                    custom_openai_config_option_names(provider_number, 'temperature'),
                    fallback=custom_openai_api_temperature,
                ),
                'max_tokens': _first_config_option_value(
                    config_parser_object,
                    'API',
                    custom_openai_config_option_names(provider_number, 'max_tokens'),
                    fallback=custom_openai_api_max_tokens,
                ),
                'top_p': _first_config_option_value(
                    config_parser_object,
                    'API',
                    custom_openai_config_option_names(provider_number, 'top_p'),
                    fallback=custom_openai_api_top_p,
                ),
                'min_p': _first_config_option_value(
                    config_parser_object,
                    'API',
                    custom_openai_config_option_names(provider_number, 'min_p'),
                    fallback=custom_openai_api_min_p,
                ),
                'api_timeout': _first_config_option_value(
                    config_parser_object,
                    'API',
                    custom_openai_config_option_names(provider_number, 'timeout'),
                    fallback=custom_openai_api_timeout,
                ),
                'api_retries': _first_config_option_value(
                    config_parser_object,
                    'API',
                    custom_openai_config_option_names(provider_number, 'retry'),
                    fallback=custom_openai_api_retries,
                ),
                'api_retry_delay': _first_config_option_value(
                    config_parser_object,
                    'API',
                    custom_openai_config_option_names(provider_number, 'retry_delay'),
                    fallback=custom_openai_api_retry_delay,
                ),
            }

        return_dict = {
            'anthropic_api': {
                'api_key': anthropic_api_key,
                'model': anthropic_model,
                'streaming': anthropic_streaming,
                'temperature': anthropic_temperature,
                'top_p': anthropic_top_p,
                'top_k': anthropic_top_k,
                'max_tokens': anthropic_max_tokens,
                'api_timeout': anthropic_api_timeout,
                'api_retries': anthropic_api_retries,
                'api_retry_delay': anthropic_api_retry_delay
            },
            'cohere_api': {
                'api_key': cohere_api_key,
                'model': cohere_model,
                'streaming': cohere_streaming,
                'temperature': cohere_temperature,
                'max_p': cohere_max_p,
                'top_k': cohere_top_k,
                'max_tokens': cohere_max_tokens,
                'api_timeout': cohere_api_timeout,
                'api_retries': cohere_api_retries,
                'api_retry_delay': cohere_api_retry_delay
            },
            'deepseek_api': {
                'api_key': deepseek_api_key,
                'model': deepseek_model,
                'streaming': deepseek_streaming,
                'temperature': deepseek_temperature,
                'top_p': deepseek_top_p,
                'min_p': deepseek_min_p,
                'max_tokens': deepseek_max_tokens,
                'api_timeout': deepseek_api_timeout,
                'api_retries': deepseek_api_retries,
                'api_retry_delay': deepseek_api_retry_delay
            },
            'qwen_api': {
                'api_key': qwen_api_key,
                'model': qwen_model,
                'streaming': qwen_streaming,
                'temperature': qwen_temperature,
                'top_p': qwen_top_p,
                'max_tokens': qwen_max_tokens,
                'api_timeout': qwen_api_timeout,
                'api_retries': qwen_api_retries,
                'api_retry_delay': qwen_api_retry_delay,
                'region': qwen_api_region,
                'api_base_url': qwen_api_base_url
            },
            'google_api': {
                'api_key': google_api_key,
                'model': google_model,
                'streaming': google_streaming,
                'temperature': google_temperature,
                'top_p': google_top_p,
                'min_p': google_min_p,
                'max_tokens': google_max_tokens,
                'api_timeout': google_api_timeout,
                'api_retries': google_api_retries,
                'api_retry_delay': google_api_retry_delay
            },
            'groq_api': {
                'api_key': groq_api_key,
                'model': groq_model,
                'streaming': groq_streaming,
                'temperature': groq_temperature,
                'top_p': groq_top_p,
                'max_tokens': groq_max_tokens,
                'api_timeout': groq_api_timeout,
                'api_retries': groq_api_retries,
                'api_retry_delay': groq_api_retry_delay
            },
            'huggingface_api': {
                'huggingface_use_router_url_format': huggingface_use_router_url_format,
                'huggingface_router_base_url': huggingface_router_base_url,
                'api_base_url': huggingface_api_base_url,
                'api_chat_path': huggingface_api_chat_path,
                'api_key': huggingface_api_key,
                'model': huggingface_model,
                'streaming': huggingface_streaming,
                'temperature': huggingface_temperature,
                'top_p': huggingface_top_p,
                'min_p': huggingface_min_p,
                'max_tokens': huggingface_max_tokens,
                'api_timeout': huggingface_api_timeout,
                'api_retries': huggingface_api_retries,
                'api_retry_delay': huggingface_api_retry_delay
            },
            'mistral_api': {
                'api_key': mistral_api_key,
                'model': mistral_model,
                'streaming': mistral_streaming,
                'temperature': mistral_temperature,
                'top_p': mistral_top_p,
                'max_tokens': mistral_max_tokens,
                'api_timeout': mistral_api_timeout,
                'api_retries': mistral_api_retries,
                'api_retry_delay': mistral_api_retry_delay
            },
            'openrouter_api': {
                'api_key': openrouter_api_key,
                'model': openrouter_model,
                'streaming': openrouter_streaming,
                'temperature': openrouter_temperature,
                'top_p': openrouter_top_p,
                'min_p': openrouter_min_p,
                'top_k': openrouter_top_k,
                'max_tokens': openrouter_max_tokens,
                'api_timeout': openrouter_api_timeout,
                'api_retries': openrouter_api_retries,
                'api_retry_delay': openrouter_api_retry_delay
            },
            'moonshot_api': {
                'api_key': moonshot_api_key,
                'model': moonshot_model,
                'api_base_url': moonshot_api_base_url,
                'streaming': moonshot_streaming,
                'temperature': moonshot_temperature,
                'top_p': moonshot_top_p,
                'max_tokens': moonshot_max_tokens,
                'api_timeout': moonshot_api_timeout,
                'api_retries': moonshot_api_retries,
                'api_retry_delay': moonshot_api_retry_delay
            },
            'zai_api': {
                'api_key': zai_api_key,
                'model': zai_model,
                'api_base_url': zai_api_base_url,
                'streaming': zai_streaming,
                'temperature': zai_temperature,
                'top_p': zai_top_p,
                'max_tokens': zai_max_tokens,
                'api_timeout': zai_api_timeout,
                'api_retries': zai_api_retries,
                'api_retry_delay': zai_api_retry_delay
            },
            'bedrock_api': {
                'api_key': bedrock_api_key,
                'region': bedrock_region,
                'runtime_endpoint': bedrock_runtime_endpoint,
                'model': bedrock_model,
                'streaming': bedrock_streaming,
                'temperature': bedrock_temperature,
                'top_p': bedrock_top_p,
                'max_tokens': bedrock_max_tokens,
                'api_timeout': bedrock_api_timeout,
                'api_retries': bedrock_api_retries,
                'api_retry_delay': bedrock_api_retry_delay
            },
            'openai_api': {
                'api_key': openai_api_key,
                'model': openai_model,
                'streaming': openai_streaming,
                'temperature': openai_temperature,
                'top_p': openai_top_p,
                'max_tokens': openai_max_tokens,
                'api_timeout': openai_api_timeout,
                'api_retries': openai_api_retries,
                'api_retry_delay': openai_api_retry_delay
            },
            'elevenlabs_api': {
                'api_key': elevenlabs_api_key,
            },
            'alltalk_api': {
                'api_ip': alltalk_api_ip,
                'default_alltalk_tts_model': default_alltalk_tts_model,
                'default_alltalk_tts_voice': default_alltalk_tts_voice,
                'default_alltalk_tts_speed': default_alltalk_tts_speed,
                'default_alltalk_tts_output_format': default_alltalk_tts_output_format,
            },
            'llama_api': {
                'api_ip': llama_api_IP,
                'api_key': llama_api_key,
                'streaming': llama_streaming,
                'temperature': llama_temperature,
                'top_p': llama_top_p,
                'min_p': llama_min_p,
                'top_k': llama_top_k,
                'max_tokens': llama_max_tokens,
                'api_timeout': llama_api_timeout,
                'api_retries': llama_api_retries,
                'api_retry_delay': llama_api_retry_delay
            },
            'ooba_api': {
                'api_ip': ooba_api_IP,
                'api_key': ooba_api_key,
                'streaming': ooba_streaming,
                'temperature': ooba_temperature,
                'top_p': ooba_top_p,
                'min_p': ooba_min_p,
                'top_k': ooba_top_k,
                'max_tokens': ooba_max_tokens,
                'api_timeout': ooba_api_timeout,
                'api_retries': ooba_api_retries,
                'api_retry_delay': ooba_api_retry_delay
            },
            'kobold_api': {
                'api_ip': kobold_api_ip,
                'api_streaming_ip': kobold_openai_api_IP,
                'api_key': kobold_api_key,
                'streaming': kobold_streaming,
                'temperature': kobold_temperature,
                'top_p': kobold_top_p,
                'top_k': kobold_top_k,
                'max_tokens': kobold_max_tokens,
                'api_timeout': kobold_api_timeout,
                'api_retries': kobold_api_retries,
                'api_retry_delay': kobold_api_retry_delay
            },
            'tabby_api': {
                'api_ip': tabby_api_IP,
                'api_key': tabby_api_key,
                'model': tabby_model,
                'streaming': tabby_streaming,
                'temperature': tabby_temperature,
                'top_p': tabby_top_p,
                'top_k': tabby_top_k,
                'min_p': tabby_min_p,
                'max_tokens': tabby_max_tokens,
                'api_timeout': tabby_api_timeout,
                'api_retries': tabby_api_retries,
                'api_retry_delay': tabby_api_retry_delay
            },
            'vllm_api': {
                'api_ip': vllm_api_url,
                'api_key': vllm_api_key,
                'model': vllm_model,
                'streaming': vllm_streaming,
                'temperature': vllm_temperature,
                'top_p': vllm_top_p,
                'top_k': vllm_top_k,
                'min_p': vllm_min_p,
                'max_tokens': vllm_max_tokens,
                'api_timeout': vllm_api_timeout,
                'api_retries': vllm_api_retries,
                'api_retry_delay': vllm_api_retry_delay
            },
            'ollama_api': {
                'api_url': ollama_api_url,
                'api_key': ollama_api_key,
                'model': ollama_model,
                'streaming': ollama_streaming,
                'temperature': ollama_temperature,
                'top_p': ollama_top_p,
                'max_tokens': ollama_max_tokens,
                'api_timeout': ollama_api_timeout,
                'api_retries': ollama_api_retries,
                'api_retry_delay': ollama_api_retry_delay
            },
            'aphrodite_api': {
                'api_ip': aphrodite_api_url,
                'api_key': aphrodite_api_key,
                'model': aphrodite_model,
                'max_tokens': aphrodite_max_tokens,
                'streaming': aphrodite_streaming,
                'api_timeout': aphrodite_api_timeout,
                'api_retries': aphrodite_api_retries,
                'api_retry_delay': aphrodite_api_retry_delay
            },
            'custom_openai_api': {
                'api_ip': custom_openai_api_ip,
                'api_key': custom_openai_api_key,
                'streaming': custom_openai_api_streaming,
                'model': custom_openai_api_model,
                'temperature': custom_openai_api_temperature,
                'max_tokens': custom_openai_api_max_tokens,
                'top_p': custom_openai_api_top_p,
                'min_p': custom_openai_api_min_p,
                'api_timeout': custom_openai_api_timeout,
                'api_retries': custom_openai_api_retries,
                'api_retry_delay': custom_openai_api_retry_delay
            },
            'custom_openai_api_2': {
                'api_ip': custom_openai2_api_ip,
                'api_key': custom_openai2_api_key,
                'streaming': custom_openai2_api_streaming,
                'model': custom_openai2_api_model,
                'temperature': custom_openai2_api_temperature,
                'max_tokens': custom_openai2_api_max_tokens,
                'top_p': custom_openai2_api_top_p,
                'min_p': custom_openai2_api_min_p,
                'api_timeout': custom_openai2_api_timeout,
                'api_retries': custom_openai2_api_retries,
                'api_retry_delay': custom_openai2_api_retry_delay
            },
            'llm_api_settings': {
                'default_api': default_api,
                'local_api_timeout': local_api_timeout,
                'local_api_retries': local_api_retries,
                'local_api_retry_delay': local_api_retry_delay,
            },
            'output_path': output_path,
            'system_preferences': {
                'save_video_transcripts': save_video_transcripts,
            },
            'processing_choice': processing_choice,
            'media_processing': {
                'max_audio_file_size_mb': max_audio_file_size_mb,
                'max_pdf_file_size_mb': max_pdf_file_size_mb,
                'max_video_file_size_mb': max_video_file_size_mb,
                'max_epub_file_size_mb': max_epub_file_size_mb,
                'max_document_file_size_mb': max_document_file_size_mb,
                'max_code_file_size_mb': max_code_file_size_mb,
                'pdf_conversion_timeout_seconds': pdf_conversion_timeout_seconds,
                'audio_processing_timeout_seconds': audio_processing_timeout_seconds,
                'video_processing_timeout_seconds': video_processing_timeout_seconds,
                'max_archive_internal_files': max_archive_internal_files,
                'max_archive_uncompressed_size_mb': max_archive_uncompressed_size_mb,
                'max_archive_nesting_depth': max_archive_nesting_depth,
                'max_archive_member_uncompressed_size_mb': max_archive_member_uncompressed_size_mb,
                'sanitize_html_uploads': sanitize_html_uploads,
                'sanitize_xml_uploads': sanitize_xml_uploads,
                'sanitize_email_html_bodies': sanitize_email_html_bodies,
                'validate_email_archive_contents': validate_email_archive_contents,
                'audio_transcription_buffer_size_mb': audio_transcription_buffer_size_mb,
                'uuid_generation_length': uuid_generation_length,
                'kept_video_max_files': kept_video_max_files,
                'kept_video_max_storage_mb': kept_video_max_storage_mb,
                'kept_video_retention_hours': kept_video_retention_hours,
            },
            'chat_dictionaries': {
                'enable_chat_dictionaries': enable_chat_dictionaries,
                'post_gen_replacement': post_gen_replacement,
                'post_gen_replacement_dict': post_gen_replacement_dict,
                'chat_dict_chat_prompts': chat_dict_chat_prompts,
                'chat_dict_RAG_prompts': chat_dict_rag_prompts,
                'chat_dict_replacement_strategy': chat_dict_replacement_strategy,
                'chat_dict_max_tokens': chat_dict_max_tokens,
                'default_rag_prompt': default_rag_prompt
            },
            'chunking_config': {
                'chunking_method': chunking_method,
                'chunk_max_size': chunk_max_size,
                'adaptive_chunking': adaptive_chunking,
                'multi_level': chunking_multi_level,
                'chunk_language': chunk_language,
                'chunk_overlap': chunk_overlap,
                'article_chunking_method': article_chunking_method,
                'article_chunk_max_size': article_chunk_max_size,
                'article_chunk_overlap': article_chunk_overlap,
                'article_adaptive_chunking': article_adaptive_chunking,
                'article_chunking_multi_level': article_chunking_multi_level,
                'article_language': article_language,
                'audio_chunking_method': audio_chunking_method,
                'audio_chunk_max_size': audio_chunk_max_size,
                'audio_chunk_overlap': audio_chunk_overlap,
                'audio_adaptive_chunking': audio_adaptive_chunking,
                'audio_chunking_multi_level': audio_chunking_multi_level,
                'audio_language': audio_language,
                'book_chunking_method': book_chunking_method,
                'book_chunk_max_size': book_chunk_max_size,
                'book_chunk_overlap': book_chunk_overlap,
                'book_adaptive_chunking': book_adaptive_chunking,
                'book_chunking_multi_level': book_chunking_multi_level,
                'book_language': book_language,
                'document_chunking_method': document_chunking_method,
                'document_chunk_max_size': document_chunk_max_size,
                'document_chunk_overlap': document_chunk_overlap,
                'document_adaptive_chunking': document_adaptive_chunking,
                'document_chunking_multi_level': document_chunking_multi_level,
                'document_language': document_language,
                'mediawiki_article_chunking_method': mediawiki_article_chunking_method,
                'mediawiki_article_chunk_max_size': mediawiki_article_chunk_max_size,
                'mediawiki_article_chunk_overlap': mediawiki_article_chunk_overlap,
                'mediawiki_article_adaptive_chunking': mediawiki_article_adaptive_chunking,
                'mediawiki_article_chunking_multi_level': mediawiki_article_chunking_multi_level,
                'mediawiki_article_language': mediawiki_article_language,
                'mediawiki_dump_chunking_method': mediawiki_dump_chunking_method,
                'mediawiki_dump_chunk_max_size': mediawiki_dump_chunk_max_size,
                'mediawiki_dump_chunk_overlap': mediawiki_dump_chunk_overlap,
                'mediawiki_dump_adaptive_chunking': mediawiki_dump_adaptive_chunking,
                'mediawiki_dump_chunking_multi_level': mediawiki_dump_chunking_multi_level,
                'mediawiki_dump_language': mediawiki_dump_language,
                'obsidian_note_chunking_method': obsidian_note_chunking_method,
                'obsidian_note_chunk_max_size': obsidian_note_chunk_max_size,
                'obsidian_note_chunk_overlap': obsidian_note_chunk_overlap,
                'obsidian_note_adaptive_chunking': obsidian_note_adaptive_chunking,
                'obsidian_note_chunking_multi_level': obsidian_note_chunking_multi_level,
                'obsidian_note_language': obsidian_note_language,
                'podcast_chunking_method': podcast_chunking_method,
                'podcast_chunk_max_size': podcast_chunk_max_size,
                'podcast_chunk_overlap': podcast_chunk_overlap,
                'podcast_adaptive_chunking': podcast_adaptive_chunking,
                'podcast_chunking_multi_level': podcast_chunking_multi_level,
                'podcast_language': podcast_language,
                'text_chunking_method': text_chunking_method,
                'text_chunk_max_size': text_chunk_max_size,
                'text_chunk_overlap': text_chunk_overlap,
                'text_adaptive_chunking': text_adaptive_chunking,
                'text_chunking_multi_level': text_chunking_multi_level,
                'text_language': text_language,
                'video_chunking_method': video_chunking_method,
                'video_chunk_max_size': video_chunk_max_size,
                'video_chunk_overlap': video_chunk_overlap,
                'video_adaptive_chunking': video_adaptive_chunking,
                'video_chunking_multi_level': video_chunking_multi_level,
                'video_language': video_language,
                # Proposition-specific
                'proposition_engine': proposition_engine,
                'proposition_prompt_profile': proposition_prompt_profile,
                'proposition_aggressiveness': proposition_aggressiveness,
                'proposition_min_proposition_length': proposition_min_proposition_length,
            },
            'embedding_config': {
                'embedding_provider': embedding_provider,
                'embedding_model': embedding_model,
                'embedding_fallback_model': embedding_fallback_model,
                'auto_generate_on_upload': auto_generate_embeddings,
                'onnx_model_path': onnx_model_path,
                'model_dir': model_dir,
                'embedding_api_url': embedding_api_url,
                'embedding_api_key': embedding_api_key,
                'chunk_size': chunk_size,
                'chunk_overlap': overlap,
                # Contextual chunking defaults for embeddings
                'enable_contextual_chunking': enable_contextual_chunking_flag,
                'contextual_llm_model': contextual_llm_model_cfg,
                'contextual_llm_provider': contextual_llm_provider_cfg,
                'contextual_llm_temperature': contextual_llm_temperature_cfg,
                'context_window_size': context_window_size_val,  # None means full-doc by default
                'context_strategy': context_strategy_val,        # auto|full|window|outline_window
                'context_token_budget': context_token_budget_val,
            },
            'auto-save': {
                'save_character_chats': save_character_chats,
                'save_rag_chats': save_rag_chats,
            },
            'default_api': default_api,
            'RAG_DEFAULT_LLM_PROVIDER': rag_default_llm_provider,
            'RAG_DEFAULT_LLM_MODEL': rag_default_llm_model,
            'local_api_timeout': local_api_timeout,
            'STT_Settings': {
                'default_stt_provider': default_stt_provider,
                'default_transcriber': default_transcriber,
                'default_batch_transcription_model': default_batch_transcription_model,
                'default_streaming_transcription_model': default_streaming_transcription_model,
                'nemo_model_variant': nemo_model_variant,
                'nemo_device': nemo_device,
                'nemo_cache_dir': nemo_cache_dir,
                'streaming_fallback_to_whisper': streaming_fallback_to_whisper,
                'parakeet_onnx_model_id': parakeet_onnx_model_id,
                'parakeet_onnx_revision': parakeet_onnx_revision,
                'mlx_model_id': mlx_model_id,
                'mlx_cache_dir': mlx_cache_dir,
                'mlx_chunk_duration': mlx_chunk_duration,
                'mlx_overlap_duration': mlx_overlap_duration,
                'buffered_chunk_duration': buffered_chunk_duration,
                'buffered_total_buffer': buffered_total_buffer,
                'buffered_merge_algo': buffered_merge_algo,
                'mlx_decoding_mode': mlx_decoding_mode,
                'mlx_beam_size': mlx_beam_size,
                'mlx_length_penalty': mlx_length_penalty,
                'mlx_patience': mlx_patience,
                'mlx_duration_reward': mlx_duration_reward,
                'mlx_sentence_max_words': mlx_sentence_max_words,
                'mlx_sentence_silence_gap': mlx_sentence_silence_gap,
                'mlx_sentence_max_duration': mlx_sentence_max_duration,
                'mlx_stream_context_left': mlx_stream_context_left,
                'mlx_stream_context_right': mlx_stream_context_right,
                'mlx_stream_depth': mlx_stream_depth,
                'mlx_stream_keep_original_attention': mlx_stream_keep_original_attention,
                # VibeVoice-ASR settings
                'vibevoice_enabled': vibevoice_enabled,
                'vibevoice_model_id': vibevoice_model_id,
                'vibevoice_device': vibevoice_device,
                'vibevoice_dtype': vibevoice_dtype,
                'vibevoice_cache_dir': vibevoice_cache_dir,
                'vibevoice_allow_download': vibevoice_allow_download,
                # Optional vLLM HTTP path for VibeVoice-ASR
                'vibevoice_vllm_enabled': vibevoice_vllm_enabled,
                'vibevoice_vllm_base_url': vibevoice_vllm_base_url,
                'vibevoice_vllm_model_id': vibevoice_vllm_model_id,
                'vibevoice_vllm_api_key': vibevoice_vllm_api_key,
                'vibevoice_vllm_timeout_seconds': vibevoice_vllm_timeout_seconds,
                # Custom vocabulary settings
                'custom_vocab_terms_file': stt_custom_vocab_terms_file,
                'custom_vocab_replacements_file': stt_custom_vocab_replacements_file,
                'custom_vocab_initial_prompt_enable': stt_custom_vocab_initial_prompt_enable,
                'custom_vocab_postprocess_enable': stt_custom_vocab_postprocess_enable,
                'custom_vocab_prompt_template': stt_custom_vocab_prompt_template,
                'custom_vocab_case_sensitive': stt_custom_vocab_case_sensitive,
                **stt_vnext_items,
            },
            # Also provide with hyphen for backward compatibility
            'STT-Settings': {
                'default_stt_provider': default_stt_provider,
                'default_transcriber': default_transcriber,
                'default_batch_transcription_model': default_batch_transcription_model,
                'default_streaming_transcription_model': default_streaming_transcription_model,
                'nemo_model_variant': nemo_model_variant,
                'nemo_device': nemo_device,
                'nemo_cache_dir': nemo_cache_dir,
                'streaming_fallback_to_whisper': streaming_fallback_to_whisper,
                'parakeet_onnx_model_id': parakeet_onnx_model_id,
                'parakeet_onnx_revision': parakeet_onnx_revision,
                'mlx_model_id': mlx_model_id,
                'mlx_cache_dir': mlx_cache_dir,
                'mlx_chunk_duration': mlx_chunk_duration,
                'mlx_overlap_duration': mlx_overlap_duration,
                'buffered_chunk_duration': buffered_chunk_duration,
                'buffered_total_buffer': buffered_total_buffer,
                'buffered_merge_algo': buffered_merge_algo,
                'mlx_decoding_mode': mlx_decoding_mode,
                'mlx_beam_size': mlx_beam_size,
                'mlx_length_penalty': mlx_length_penalty,
                'mlx_patience': mlx_patience,
                'mlx_duration_reward': mlx_duration_reward,
                'mlx_sentence_max_words': mlx_sentence_max_words,
                'mlx_sentence_silence_gap': mlx_sentence_silence_gap,
                'mlx_sentence_max_duration': mlx_sentence_max_duration,
                'mlx_stream_context_left': mlx_stream_context_left,
                'mlx_stream_context_right': mlx_stream_context_right,
                'mlx_stream_depth': mlx_stream_depth,
                'mlx_stream_keep_original_attention': mlx_stream_keep_original_attention,
                # VibeVoice-ASR settings
                'vibevoice_enabled': vibevoice_enabled,
                'vibevoice_model_id': vibevoice_model_id,
                'vibevoice_device': vibevoice_device,
                'vibevoice_dtype': vibevoice_dtype,
                'vibevoice_cache_dir': vibevoice_cache_dir,
                'vibevoice_allow_download': vibevoice_allow_download,
                # Optional vLLM HTTP path for VibeVoice-ASR
                'vibevoice_vllm_enabled': vibevoice_vllm_enabled,
                'vibevoice_vllm_base_url': vibevoice_vllm_base_url,
                'vibevoice_vllm_model_id': vibevoice_vllm_model_id,
                'vibevoice_vllm_api_key': vibevoice_vllm_api_key,
                'vibevoice_vllm_timeout_seconds': vibevoice_vllm_timeout_seconds,
                # Custom vocabulary settings
                'custom_vocab_terms_file': stt_custom_vocab_terms_file,
                'custom_vocab_replacements_file': stt_custom_vocab_replacements_file,
                'custom_vocab_initial_prompt_enable': stt_custom_vocab_initial_prompt_enable,
                'custom_vocab_postprocess_enable': stt_custom_vocab_postprocess_enable,
                'custom_vocab_prompt_template': stt_custom_vocab_prompt_template,
                'custom_vocab_case_sensitive': stt_custom_vocab_case_sensitive,
                **stt_vnext_items,
            },
            'diarization': diarization_config,
            'tts_settings': {
                'default_tts_provider': default_tts_provider,
                'tts_voice': tts_voice,
                'local_tts_device': local_tts_device,
                'tts_history_enabled': tts_history_enabled,
                'tts_history_store_text': tts_history_store_text,
                'tts_history_store_failed': tts_history_store_failed,
                'tts_history_hash_key': tts_history_hash_key,
                'tts_history_retention_days': tts_history_retention_days,
                'tts_history_max_rows_per_user': tts_history_max_rows_per_user,
                'tts_history_purge_interval_hours': tts_history_purge_interval_hours,
                # OpenAI
                'default_openai_tts_voice': default_openai_tts_voice,
                'default_openai_tts_speed': default_openai_tts_speed,
                'default_openai_tts_model': default_openai_tts_model,
                'default_openai_tts_output_format': default_openai_tts_output_format,
                # Google
                'default_google_tts_model': default_google_tts_model,
                'default_google_tts_voice': default_google_tts_voice,
                'default_google_tts_speed': default_google_tts_speed,
                # ElevenLabs
                'default_eleven_tts_model': default_eleven_tts_model,
                'default_eleven_tts_voice': default_eleven_tts_voice,
                'default_eleven_tts_language_code': default_eleven_tts_language_code,
                'default_eleven_tts_voice_stability': default_eleven_tts_voice_stability,
                'default_eleven_tts_voice_similiarity_boost': default_eleven_tts_voice_similiarity_boost,
                'default_eleven_tts_voice_style': default_eleven_tts_voice_style,
                'default_eleven_tts_voice_use_speaker_boost': default_eleven_tts_voice_use_speaker_boost,
                'default_eleven_tts_output_format': default_eleven_tts_output_format,
                # Open Source / Self-Hosted TTS
                # GPT SoVITS
                # 'default_gpt_tts_model': default_gpt_tts_model,
                # 'default_gpt_tts_voice': default_gpt_tts_voice,
                # 'default_gpt_tts_speed': default_gpt_tts_speed,
                # 'default_gpt_tts_output_format': default_gpt_tts_output_format
                # AllTalk
                'alltalk_api_ip': alltalk_api_ip,
                'default_alltalk_tts_model': default_alltalk_tts_model,
                'default_alltalk_tts_voice': default_alltalk_tts_voice,
                'default_alltalk_tts_speed': default_alltalk_tts_speed,
                'default_alltalk_tts_output_format': default_alltalk_tts_output_format,
                # Kokoro
                'default_kokoro_tts_model': default_kokoro_tts_model,
                'default_kokoro_tts_voice': default_kokoro_tts_voice,
                'default_kokoro_tts_speed': default_kokoro_tts_speed,
                'default_kokoro_tts_output_format': default_kokoro_tts_output_format,
                # Self-hosted OpenAI API
                'default_openai_api_tts_model': default_openai_api_tts_model,
                'default_openai_api_tts_voice': default_openai_api_tts_voice,
                'default_openai_api_tts_speed': default_openai_api_tts_speed,
                'default_openai_api_tts_output_format': default_openai_api_tts_output_format,
                'default_openai_api_tts_streaming': default_openai_api_tts_streaming,
            },
            'search_settings': {
                'default_search_provider': search_provider_default,
                'search_language_query': search_language_query,
                'search_language_results': search_language_results,
                'search_language_analysis': search_language_analysis,
                'search_default_max_queries': search_default_max_queries,
                'search_enable_subquery': search_enable_subquery,
                'search_enable_subquery_count_max': search_enable_subquery_count_max,
                'search_result_rerank': search_result_rerank,
                'search_result_max': search_result_max,
                'search_result_max_per_query': search_result_max_per_query,
                'search_result_blacklist': search_result_blacklist,
                'search_result_display_type': search_result_display_type,
                'search_result_display_metadata': search_result_display_metadata,
                'search_result_save_to_db': search_result_save_to_db,
                'search_result_analysis_tone': search_result_analysis_tone,
                'relevance_analysis_llm': relevance_analysis_llm,
                'final_answer_llm': final_answer_llm,
            },
            'search_engines': {
                'baidu_search_api_key': baidu_search_api_key,
                'bing_search_api_key': bing_search_api_key,
                'bing_country_code': bing_country_code,
                'bing_search_api_url': bing_search_api_url,
                'brave_search_api_key': brave_search_api_key,
                'brave_search_ai_api_key': brave_search_ai_api_key,
                'brave_country_code': brave_country_code,
                'duckduckgo_search_api_key': duckduckgo_search_api_key,
                'google_search_api_url': google_search_api_url,
                'google_search_api_key': google_search_api_key,
                'google_search_engine_id': google_search_engine_id,
                'google_simp_trad_chinese': google_simp_trad_chinese,
                'limit_google_search_to_country': limit_google_search_to_country,
                'google_search_country': google_search_country,
                'google_search_country_code': google_search_country_code,
                'google_search_filter_setting': google_filter_setting,
                'google_user_geolocation': google_user_geolocation,
                'google_ui_language': google_ui_language,
                'google_limit_search_results_to_language': google_limit_search_results_to_language,
                'google_site_search_include': google_site_search_include,
                'google_site_search_exclude': google_site_search_exclude,
                'google_sort_results_by': google_sort_results_by,
                'google_default_search_results': google_default_search_results,
                'google_safe_search': google_safe_search,
                'google_enable_site_search' : google_enable_site_search,
                'kagi_search_api_key': kagi_search_api_key,
                'searx_search_api_url': search_engine_searx_api,
                'tavily_search_api_key': tavily_search_api_key,
                'serper_search_api_key': serper_search_api_key,
                'serper_search_api_url': serper_search_api_url,
                'exa_search_api_key': exa_search_api_key,
                'exa_search_api_url': exa_search_api_url,
                'firecrawl_api_key': firecrawl_api_key,
                'firecrawl_search_api_url': firecrawl_search_api_url,
                'yandex_search_api_key': yandex_search_api_key,
                'yandex_search_engine_id': yandex_search_engine_id
            },
            'prompts': {
                'sub_question_generation_prompt': sub_question_generation_prompt,
                'search_result_relevance_eval_prompt': search_result_relevance_eval_prompt,
                'analyze_search_results_prompt': analyze_search_results_prompt,
            },
            'web_scraper':{
                'web_scraper_api_key': web_scraper_api_key,
                'web_scraper_api_url': web_scraper_api_url,
                'web_scraper_api_timeout': web_scraper_api_timeout,
                'web_scraper_api_retries': web_scraper_api_retries,
                'web_scraper_api_retry_delay': web_scraper_api_retry_delay,
                'web_scraper_retry_count': web_scraper_retry_count,
            'web_scraper_retry_timeout': web_scraper_retry_timeout,
            'web_scraper_stealth_playwright': web_scraper_stealth_playwright,
            'custom_scrapers_yaml_path': custom_scrapers_yaml_path,
            'web_scraper_default_backend': web_scraper_default_backend,
            'web_scraper_ua_mode': web_scraper_ua_mode,
            # Crawl feature flags
            'web_crawl_strategy': web_crawl_strategy,
            'web_crawl_include_external': web_crawl_include_external,
            'web_crawl_score_threshold': web_crawl_score_threshold,
            'web_crawl_max_pages': web_crawl_max_pages,
            'web_crawl_allowed_domains': web_crawl_allowed_domains,
            'web_crawl_blocked_domains': web_crawl_blocked_domains,
            'web_scraper_respect_robots': web_scraper_respect_robots,
            'web_outbound_policy_mode': web_outbound_policy_mode_value,
            # Scorers
            'web_crawl_enable_keyword_scorer': web_crawl_enable_keyword,
            'web_crawl_keywords': web_crawl_keywords,
            'web_crawl_enable_domain_map': web_crawl_enable_domain_map,
            'web_crawl_domain_map': web_crawl_domain_map,
            },
            'Redis': dict(config_parser_object.items('Redis')) if config_parser_object.has_section('Redis') else {},
            'Web-Scraping': dict(config_parser_object.items('Web-Scraping')) if config_parser_object.has_section('Web-Scraping') else {}
        }
        for provider_number in iter_custom_openai_provider_numbers(start=3):
            numbered_config = _numbered_custom_openai_config(provider_number)
            if numbered_config is not None:
                return_dict[custom_openai_section_name(provider_number)] = numbered_config

        # Assemble minimal RAG config section (vector store + pgvector params)
        try:
            rag_section = {}
            if config_parser_object.has_section('RAG'):
                rag_section['vector_store_type'] = config_parser_object.get('RAG', 'vector_store_type', fallback='chromadb')
                rag_section['distance_metric'] = config_parser_object.get('RAG', 'distance_metric', fallback='cosine')
                rag_section['collection_prefix'] = config_parser_object.get('RAG', 'collection_prefix', fallback='unified')
                # PGVector connection params under [RAG]
                rag_section['pgvector'] = {
                    'host': config_parser_object.get('RAG', 'pgvector_host', fallback='localhost'),
                    'port': config_parser_object.getint('RAG', 'pgvector_port', fallback=5432),
                    'database': config_parser_object.get('RAG', 'pgvector_database', fallback='postgres'),
                    'user': config_parser_object.get('RAG', 'pgvector_user', fallback='postgres'),
                    'password': config_parser_object.get('RAG', 'pgvector_password', fallback=''),
                    'sslmode': config_parser_object.get('RAG', 'pgvector_sslmode', fallback='prefer'),
                    'dsn': (config_parser_object.get('RAG', 'pgvector_dsn', fallback='') or None),
                    # Pool configuration (psycopg_pool)
                    'pool_min_size': config_parser_object.getint('RAG', 'pgvector_pool_min_size', fallback=1),
                    'pool_max_size': config_parser_object.getint('RAG', 'pgvector_pool_max_size', fallback=5),
                    'pool_size': config_parser_object.getint('RAG', 'pgvector_pool_size', fallback=5),
                    # HNSW tuning
                    'hnsw_ef_search': config_parser_object.getint('RAG', 'pgvector_hnsw_ef_search', fallback=64),
                }
            return_dict['RAG'] = rag_section
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            # Non-fatal: keep defaults
            pass

        # Optional OCR section for backend preferences and defaults
        try:
            if config_parser_object.has_section('OCR'):
                ocr_section = {}
                backend_priority = config_parser_object.get('OCR', 'backend_priority', fallback='')
                if backend_priority:
                    ocr_section['backend_priority'] = backend_priority
                page_conc = config_parser_object.get('OCR', 'page_concurrency_default', fallback='')
                if page_conc:
                    ocr_section['page_concurrency_default'] = int(page_conc)
                sglang_timeout = config_parser_object.get('OCR', 'sglang_timeout', fallback='')
                if sglang_timeout:
                    ocr_section['sglang_timeout'] = int(sglang_timeout)
                if ocr_section:
                    return_dict['OCR'] = ocr_section
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            pass

        return return_dict
    except _CONFIG_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Error loading config: {e!s}")
        return None


# --- Lazy Configuration Proxies ---

class _LazyMapping(MutableMapping[str, Any]):
    """MutableMapping proxy that materializes its data on first access."""

    __slots__ = ("_loader", "_data")

    def __init__(self, loader):
        object.__setattr__(self, "_loader", loader)
        object.__setattr__(self, "_data", None)

    def _ensure(self):
        if object.__getattribute__(self, "_data") is None:
            loader = object.__getattribute__(self, "_loader")
            data = loader()
            if data is None:
                data = {}
            object.__setattr__(self, "_data", data)

    def __getitem__(self, key):
        self._ensure()
        return object.__getattribute__(self, "_data")[key]

    def __setitem__(self, key, value):
        self._ensure()
        object.__getattribute__(self, "_data")[key] = value

    def __delitem__(self, key):
        self._ensure()
        del object.__getattribute__(self, "_data")[key]

    def __iter__(self):
        self._ensure()
        return iter(object.__getattribute__(self, "_data"))

    def __len__(self):
        self._ensure()
        return len(object.__getattribute__(self, "_data"))

    def get(self, key, default=None):
        self._ensure()
        return object.__getattribute__(self, "_data").get(key, default)

    def keys(self):
        self._ensure()
        return object.__getattribute__(self, "_data").keys()

    def values(self):
        self._ensure()
        return object.__getattribute__(self, "_data").values()

    def items(self):
        self._ensure()
        return object.__getattribute__(self, "_data").items()

    def pop(self, key, default=None):
        self._ensure()
        return object.__getattribute__(self, "_data").pop(key, default)

    def update(self, *args, **kwargs):
        self._ensure()
        object.__getattribute__(self, "_data").update(*args, **kwargs)

    def clear(self):
        self._ensure()
        object.__getattribute__(self, "_data").clear()

    def __contains__(self, item):
        self._ensure()
        return item in object.__getattribute__(self, "_data")


class LazySettings(_LazyMapping):
    """Lazy settings mapping that also supports attribute-style access."""

    def __getattr__(self, name):
        if name in {"_loader", "_data"}:
            return object.__getattribute__(self, name)
        self._ensure()
        data = object.__getattribute__(self, "_data")
        if hasattr(data, name):
            return getattr(data, name)
        try:
            return data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        if name in {"_loader", "_data"}:
            object.__setattr__(self, name, value)
            return
        self._ensure()
        object.__getattribute__(self, "_data")[name] = value

    def __delattr__(self, name):
        if name in {"_loader", "_data"}:
            raise AttributeError("Cannot delete internal attribute")
        self._ensure()
        data = object.__getattribute__(self, "_data")
        try:
            del data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __dir__(self):
        self._ensure()
        data = object.__getattribute__(self, "_data")
        return sorted(set(super().__dir__()) | set(data.keys()))


class LazyConfigData(_LazyMapping):
    """Lazy loader for the comprehensive configuration dictionary."""

    pass


default_api_endpoint = "openai"


def _settings_loader():
    return load_settings()


def _config_loader():
    data = load_and_log_configs()
    if data is None:
        data = {}
    global default_api_endpoint
    default_api_endpoint = data.get('default_api', 'openai')
    return data


settings = LazySettings(_settings_loader)
config = settings
loaded_config_data = LazyConfigData(_config_loader)
_LEGACY_SENTINEL = object()


def load_all_sections_for_test() -> "ConfigSections":
    from tldw_Server_API.app.core.config_sections import load_config_sections

    return load_config_sections(load_comprehensive_config())


def legacy_get(key: str, default: Any = None) -> Any:
    """Compatibility accessor for legacy call sites during modular config migration."""
    if not isinstance(key, str) or not key.strip():
        return default

    for candidate in (key, key.upper(), key.lower()):
        try:
            value = settings.get(candidate, _LEGACY_SENTINEL)
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            value = _LEGACY_SENTINEL
        if value is not _LEGACY_SENTINEL:
            return value

    if "." in key:
        section, _, field = key.partition(".")
        section_data = loaded_config_data.get(section, {})
        if isinstance(section_data, MutableMapping):
            for candidate in (field, field.lower(), field.upper()):
                if candidate in section_data:
                    return section_data[candidate]
        value = get_config_value(section, field, default=_LEGACY_SENTINEL)
        if value is not _LEGACY_SENTINEL:
            return value

    return default

def get_stt_config() -> dict[str, Any]:
    """
    Return the canonical STT configuration export as a plain dict.

    This helper centralizes resolution of STT-related settings so callers do
    not need to worry about whether `loaded_config_data` is a lazy wrapper,
    a plain dict, or needs to be loaded via `load_and_log_configs()`.
    """
    cfg: Any = loaded_config_data
    try:
        cfg = cfg() if callable(cfg) else cfg
    except _CONFIG_NONCRITICAL_EXCEPTIONS:
        cfg = None

    if not cfg:
        try:
            cfg = load_and_log_configs() or {}
        except _CONFIG_NONCRITICAL_EXCEPTIONS:
            cfg = {}

    if not isinstance(cfg, MutableMapping):
        return {}

    stt_section = cfg.get("STT_Settings")
    if not isinstance(stt_section, MutableMapping):
        stt_section = cfg.get("STT-Settings")
    return dict(stt_section) if isinstance(stt_section, MutableMapping) else {}

_LOGGER_READY = True
_flush_startup_logs()


def refresh_config_cache() -> None:
    """Refresh cached config.txt data (for tests or dynamic reloads)."""
    global _CONFIG_PARSER_CACHE
    _CONFIG_PARSER_CACHE = None
    _CONFIG_SOURCE_METADATA.update({"path": None, "loaded": False})
    load_comprehensive_config.cache_clear()
    _route_toggle_policy.cache_clear()
    should_disable_cors.cache_clear()
    should_allow_cors_credentials.cache_clear()
    is_production_environment.cache_clear()


def clear_config_cache() -> None:
    """Clear cached configuration loaders (for tests or dynamic reloads)."""
    refresh_config_cache()
    global default_api_endpoint
    default_api_endpoint = "openai"
    object.__setattr__(settings, "_data", None)
    object.__setattr__(loaded_config_data, "_data", None)
# ---------------------------------------------------------------------------
# Startup config validation
# ---------------------------------------------------------------------------

_PLACEHOLDER_LITERALS = frozenset({
    "FIXME", "TODO", "TBD", "CHANGE_ME", "CHANGE-ME",
    "PLACEHOLDER", "NONE", "NULL", "N/A", "NA",
})


def validate_config() -> list[str]:
    """Validate the loaded configuration and return a list of warnings.

    Call this during lifespan startup. Warnings are logged but do not
    prevent startup. Returns the list so tests can assert on it.
    """
    warnings: list[str] = []
    cfg = dict(loaded_config_data)

    def _iter_scalar_values(prefix: str, value: Any):
        if isinstance(value, MutableMapping):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from _iter_scalar_values(child_prefix, child)
            return
        if isinstance(value, list | tuple):
            for index, child in enumerate(value):
                child_prefix = f"{prefix}[{index}]"
                yield from _iter_scalar_values(child_prefix, child)
            return
        yield prefix, value

    validation_values = dict(_iter_scalar_values("", cfg))
    validation_values.update({
        "Database.pg_connection_string": get_config_value("Database", "pg_connection_string", default=""),
        "Image-Generation.swarmui_base_url": get_config_value("Image-Generation", "swarmui_base_url", default=""),
    })

    # Check for placeholder values in any config key
    for key, value in validation_values.items():
        if isinstance(value, str) and value.strip().upper() in _PLACEHOLDER_LITERALS:
            msg = f"Config key '{key}' has placeholder value '{value}' — set a real value or leave empty"
            warnings.append(msg)

    # Check critical URL values parse correctly
    url_rules = {
        "embedding_config.embedding_api_url": ("http://", "https://"),
        "Database.pg_connection_string": ("postgres://", "postgresql://", "postgres+", "postgresql+"),
        "Image-Generation.swarmui_base_url": ("http://", "https://"),
    }
    for key, allowed_prefixes in url_rules.items():
        val = validation_values.get(key, "")
        if val and not isinstance(val, str):
            warnings.append(f"Config key '{key}' should be a string, got {type(val).__name__}")
        elif isinstance(val, str) and val and not val.startswith(allowed_prefixes):
            warnings.append(f"Config key '{key}' has unexpected URL scheme")

    for w in warnings:
        logger.warning("Config validation: {}", w)

    if not warnings:
        logger.info("Config validation passed — no issues found")

    return warnings


# --- Optional: Export individual variables if needed for backward compatibility (less recommended) ---
# SINGLE_USER_MODE = settings["SINGLE_USER_MODE"]
# SINGLE_USER_FIXED_ID = settings["SINGLE_USER_FIXED_ID"]
# ... etc ...
#
# End of config.py
#######################################################################################################################
