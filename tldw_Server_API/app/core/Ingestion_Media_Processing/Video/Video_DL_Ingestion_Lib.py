# Video_DL_Ingestion_Lib.py
#########################################
# Video Downloader and Ingestion Library
# This library is used to handle downloading videos from YouTube and other platforms.
# It also handles the ingestion of the videos into the database.
# It uses yt-dlp to extract video information and download the videos.
####
import json

####################
# Function List
#
# 1. get_video_info(url)
# 2. create_download_directory(title)
# 3. sanitize_filename(title)
# 4. normalize_title(title)
# 5. get_youtube(video_url)
# 6. get_playlist_videos(playlist_url)
# 7. download_video(video_url, download_path, info_dict, download_video_flag)
# 8. save_to_file(video_urls, filename)
# 9. save_summary_to_file(summary, file_path)
# 10. process_url(url, num_speakers, whisper_model, custom_prompt, offset, api_name, api_key, vad_filter, download_video, download_audio, rolling_summarization, detail_level, question_box, keywords, chunk_summarization, chunk_duration_input, words_per_second_input)
#
#
####################
# Import necessary libraries to run solo for testing
import asyncio
import os
import re
import shutil
import tempfile
import time

#
# 3rd-Party Imports
import unicodedata
import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse, urlunparse

import yt_dlp
from loguru import logger

# Import Local
from tldw_Server_API.app.core.Evaluations.ms_g_eval import run_geval


# Lazy import for transcription to avoid heavy dependencies at module import time
def perform_transcription(*args, **kwargs):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
        perform_transcription as _perform_transcription,
    )
    return _perform_transcription(*args, **kwargs)
from tldw_Server_API.app.core.Chunking import improved_chunking_process
from tldw_Server_API.app.core.config import loaded_config_data
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_api_key_env_keys,
    custom_openai_provider_number,
    custom_openai_section_name,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import resolve_safe_local_path
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.Security.egress import evaluate_url_policy
from tldw_Server_API.app.core.Utils.Utils import convert_to_seconds, extract_text_from_segments, logging

#
#######################################################################################################################
# Function Definitions
#

# ffmpeg check
try:
    # Adjust .parent calls based on your actual structure to reach the project root
    # Example: If this file is in app/core/Ingestion/Video/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
except NameError: # Fallback if __file__ is not defined
    PROJECT_ROOT = Path(os.getcwd())
    logging.warning(f"Could not determine project root from __file__, falling back to CWD: {PROJECT_ROOT}")


_PROVIDER_SECTION_MAP: dict[str, str] = {
    "openai": "openai_api",
    "anthropic": "anthropic_api",
    "cohere": "cohere_api",
    "groq": "groq_api",
    "deepseek": "deepseek_api",
    "mistral": "mistral_api",
    "openrouter": "openrouter_api",
    "novita": "novita_api",
    "poe": "poe_api",
    "together": "together_api",
    "huggingface": "huggingface_api",
    "google": "google_api",
    "qwen": "qwen_api",
    "custom-openai-api": "custom_openai_api",
    "custom-openai-api-2": "custom_openai_api_2",
    "llama.cpp": "llama_api",
    "kobold": "kobold_api",
    "ooba": "ooba_api",
    "tabbyapi": "tabby_api",
    "vllm": "vllm_api",
    "local-llm": "local_llm",
    "ollama": "ollama_api",
    "aphrodite": "aphrodite_api",
}

_PROVIDER_ENV_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "COHERE_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "novita": "NOVITA_API_KEY",
    "poe": "POE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "google": "GOOGLE_API_KEY",
    "qwen": "QWEN_API_KEY",
    "custom-openai-api": "CUSTOM_OPENAI_API_KEY",
    "custom-openai-api-2": "CUSTOM_OPENAI2_API_KEY",
    "llama.cpp": "LLAMA_CPP_API_KEY",
    "kobold": "KOBOLD_API_KEY",
    "ooba": "OOBA_API_KEY",
    "tabbyapi": "TABBYAPI_API_KEY",
    "vllm": "VLLM_API_KEY",
    "local-llm": "LOCAL_LLM_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "aphrodite": "APHRODITE_API_KEY",
}

_PROVIDERS_REQUIRING_KEYS = {
    "openai",
    "anthropic",
    "cohere",
    "groq",
    "openrouter",
    "deepseek",
    "huggingface",
    "mistral",
    "google",
    "qwen",
    "custom-openai-api",
    "custom-openai-api-2",
    "aphrodite",
}

media_config = loaded_config_data.get('media_processing', {}) if loaded_config_data else {}
DEFAULT_MAX_VIDEO_FILE_SIZE_MB = media_config.get('max_video_file_size_mb', 1000)
DEFAULT_MAX_VIDEO_FILE_SIZE_BYTES = DEFAULT_MAX_VIDEO_FILE_SIZE_MB * 1024 * 1024
_TRANSCRIPTION_EXTRACTION_ERROR_SENTINEL = "Error: Unable to extract transcription"
_KEEP_VIDEO_MAX_FILES = max(0, int(media_config.get('kept_video_max_files', 5)))  # per-user file cap
_KEEP_VIDEO_MAX_STORAGE_MB = max(0, int(media_config.get('kept_video_max_storage_mb', 500)))
_KEEP_VIDEO_MAX_BYTES = _KEEP_VIDEO_MAX_STORAGE_MB * 1024 * 1024
_KEEP_VIDEO_RETENTION_SECONDS = max(0, int(media_config.get('kept_video_retention_hours', 2))) * 3600
_VIDEO_STORAGE_ROOT = Path(tempfile.gettempdir()) / "tldw_kept_videos"

_VIDEO_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    json.JSONDecodeError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


def _validate_downloaded_url_video_file(downloaded_path: Path) -> None:
    """
    Apply upload-equivalent validation to URL-downloaded video payloads.
    """
    from tldw_Server_API.app.core.Ingestion_Media_Processing.persistence import (  # type: ignore  # noqa: E501
        _validate_downloaded_url_file,
    )

    _validate_downloaded_url_file(
        downloaded_path=downloaded_path,
        processing_filename=downloaded_path.name,
        media_type="video",
        form_data=None,
        media_mod=None,
        allowed_extensions=None,
    )


def _enforce_download_quota(user_id: Optional[int], downloaded_path: Path) -> None:
    """Raise ValueError when a downloaded URL payload would exceed storage quota."""
    if user_id is None:
        return

    from tldw_Server_API.app.services.storage_quota_service import get_storage_quota_service

    async def _check_quota() -> None:
        quota_service = get_storage_quota_service()
        size_bytes = downloaded_path.stat().st_size
        has_quota, info = await quota_service.check_quota(
            user_id,
            size_bytes,
            raise_on_exceed=False,
        )
        if has_quota:
            return
        raise ValueError(
            "Storage quota exceeded. Current: "
            f"{info['current_usage_mb']}MB, "
            f"New: {info['new_size_mb']}MB, "
            f"Quota: {info['quota_mb']}MB, "
            f"Available: {info['available_mb']}MB"
        )

    asyncio.run(_check_quota())


def _safe_remove_file(file_path: Path) -> None:
    """Attempt to remove a file while swallowing non-critical errors."""
    try:
        if file_path.exists():
            file_path.unlink()
            logging.debug(f"Removed stored video: {file_path}")
    except _VIDEO_NONCRITICAL_EXCEPTIONS as exc:
        logging.warning(f"Failed to remove stored video '{file_path}': {exc}")


def _collect_storage_entries(storage_dir: Path) -> list[dict[str, Any]]:
    """Return storage entries sorted by modification time (oldest first)."""
    entries: list[dict[str, Any]] = []
    if not storage_dir.exists():
        return entries
    for item in storage_dir.iterdir():
        if not item.is_file():
            continue
        try:
            stat = item.stat()
        except OSError as exc:
            logging.warning(f"Could not stat stored video '{item}': {exc}")
            continue
        entries.append({"path": item, "mtime": stat.st_mtime, "size": stat.st_size})
    entries.sort(key=lambda entry: entry["mtime"])
    return entries


def _purge_expired_videos(storage_dir: Path, retention_seconds: int) -> None:
    """Remove stored videos older than the retention period."""
    if retention_seconds <= 0 or not storage_dir.exists():
        return
    expiry_threshold = time.time() - retention_seconds
    for entry in _collect_storage_entries(storage_dir):
        if entry["mtime"] < expiry_threshold:
            logging.debug(f"Purging expired stored video: {entry['path']}")
            _safe_remove_file(entry["path"])


def _enforce_storage_limits(storage_dir: Path, max_files: int, max_bytes: int) -> None:
    """Ensure stored videos respect configured count and size limits."""
    if not storage_dir.exists():
        return
    entries = _collect_storage_entries(storage_dir)
    total_bytes = sum(entry["size"] for entry in entries)

    def _needs_prune() -> bool:
        file_limit_exceeded = max_files > 0 and len(entries) > max_files
        size_limit_exceeded = max_bytes > 0 and total_bytes > max_bytes
        return file_limit_exceeded or size_limit_exceeded

    while entries and _needs_prune():
        oldest = entries.pop(0)
        logging.debug(f"Storage limit exceeded, removing oldest video: {oldest['path']}")
        _safe_remove_file(oldest["path"])
        total_bytes -= oldest["size"]


def _ensure_storage_dir(user_id: Optional[int]) -> Path:
    """Return the per-user storage directory, creating it on demand."""
    user_segment = str(user_id) if user_id is not None else "default"
    storage_dir = _VIDEO_STORAGE_ROOT / user_segment
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def _store_video_file(source_path: Path, user_id: Optional[int]) -> Optional[Path]:
    """
    Persist a copy of the downloaded video for later retrieval, honoring limits.

    Returns the stored path on success, or None when storage is disabled or fails.
    """
    def _coerce_limit(value: Optional[int]) -> int:
        """Convert limit settings (possibly str/None) into a non-negative int."""
        try:
            return max(0, int(value))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0

    max_files = _coerce_limit(_KEEP_VIDEO_MAX_FILES)
    storage_mb_limit = _coerce_limit(_KEEP_VIDEO_MAX_STORAGE_MB)
    bytes_limit_setting = _coerce_limit(_KEEP_VIDEO_MAX_BYTES)

    # Derive an effective byte ceiling using both MB and explicit byte overrides.
    storage_bytes_limit = storage_mb_limit * 1024 * 1024 if storage_mb_limit else 0
    if storage_bytes_limit and bytes_limit_setting:
        effective_max_bytes = min(storage_bytes_limit, bytes_limit_setting)
    else:
        effective_max_bytes = storage_bytes_limit or bytes_limit_setting

    if max_files <= 0 or effective_max_bytes <= 0:
        logging.info("Kept-video storage disabled by configuration; skipping persistence.")
        return None

    if not source_path.exists():
        logging.warning(f"Cannot keep original video; source path missing: {source_path}")
        return None

    try:
        source_size = source_path.stat().st_size
    except OSError as exc:
        logging.error(f"Unable to read size for stored video '{source_path}': {exc}")
        return None

    if source_size > effective_max_bytes:
        logging.warning(
            f"Skipping kept-video storage for '{source_path.name}': "
            f"{source_size / (1024 * 1024):.2f}MB exceeds configured limit "
            f"({effective_max_bytes / (1024 * 1024):.2f}MB)."
        )
        return None

    storage_dir = _ensure_storage_dir(user_id)
    _purge_expired_videos(storage_dir, _KEEP_VIDEO_RETENTION_SECONDS)

    destination = storage_dir / source_path.name
    if destination.exists():
        destination = storage_dir / f"{source_path.stem}_{uuid.uuid4().hex[:6]}{source_path.suffix}"

    try:
        shutil.copy2(source_path, destination)
        logging.info(f"Stored kept video copy at {destination}")
    except _VIDEO_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
        logging.exception(f"Failed to copy video '{source_path}' into kept storage: {exc}")
        return None

    _enforce_storage_limits(storage_dir, max_files, effective_max_bytes)
    return destination


def _resolve_eval_api_key(api_name: Optional[str]) -> Optional[str]:
    """
    Resolve an API key for evaluation providers from loaded configuration or environment.
    """
    if not api_name:
        return None
    provider = api_name.lower().strip()
    custom_number = custom_openai_provider_number(provider)
    section_key = (
        custom_openai_section_name(custom_number)
        if custom_number is not None
        else _PROVIDER_SECTION_MAP.get(
            provider,
            f"{provider.replace('-', '_').replace('.', '_')}_api",
        )
    )
    normalized_env_name = provider.upper().replace("-", "_").replace(".", "_")
    env_keys = (
        custom_openai_api_key_env_keys(custom_number)
        if custom_number is not None
        else (
            _PROVIDER_ENV_MAP.get(
                provider,
                f"{''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in normalized_env_name)}_API_KEY",
            ),
        )
    )

    api_key: Optional[Any] = None
    try:
        provider_section = loaded_config_data.get(section_key, {}) if loaded_config_data else {}
        if isinstance(provider_section, dict):
            api_key = provider_section.get("api_key")
    except _VIDEO_NONCRITICAL_EXCEPTIONS:
        api_key = None

    if not api_key:
        for env_key in env_keys:
            api_key = os.getenv(env_key)
            if api_key:
                break

    return str(api_key) if api_key else None


def _extract_declared_filesize(info_dict: Optional[dict[str, Any]]) -> Optional[int]:
    """Best-effort extraction of declared file size from yt-dlp metadata."""
    if not isinstance(info_dict, dict):
        return None

    candidate_keys = (
        "filesize",
        "filesize_approx",
        "filesize_before_download",
        "filesize_after_download",
    )
    for key in candidate_keys:
        size_val = info_dict.get(key)
        if size_val is None:
            continue
        try:
            return int(size_val)
        except (TypeError, ValueError):
            continue

    # Check nested format entries
    for table_key in ("requested_formats", "formats"):
        for fmt in info_dict.get(table_key, []) or []:
            for key in ("filesize", "filesize_approx"):
                size_val = fmt.get(key)
                if size_val is None:
                    continue
                try:
                    return int(size_val)
                except (TypeError, ValueError):
                    continue
    return None

def normalize_title(title):
    # Normalize the string to 'NFKD' form and encode to 'ascii' ignoring non-ascii characters
    title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')
    title = title.replace('/', '_').replace('\\', '_').replace(':', '_').replace('"', '').replace('*', '').replace('?',
                                                                                                                   '').replace(
        '<', '').replace('>', '').replace('|', '')
    return title

def get_video_info(url: str, *, use_cookies: bool = False, cookies: Optional[dict[str, Any] | str] = None) -> dict:
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }
    if use_cookies and cookies:
        try:
            cookie_header = _cookies_to_header_value(cookies)
            if cookie_header:
                ydl_opts.setdefault('http_headers', {})['Cookie'] = cookie_header
        except _VIDEO_NONCRITICAL_EXCEPTIONS:
            pass
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(url, download=False)
            return info_dict
        except _VIDEO_NONCRITICAL_EXCEPTIONS as e:
            logging.error(f"Error extracting video info: {e}")
            return None


def get_youtube(video_url: str, *, use_cookies: bool = False, cookies: Optional[dict[str, Any] | str] = None):
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',
        'noplaylist': False,
        'quiet': True,
        'extract_flat': True
    }
    if use_cookies and cookies:
        try:
            cookie_header = _cookies_to_header_value(cookies)
            if cookie_header:
                ydl_opts.setdefault('http_headers', {})['Cookie'] = cookie_header
        except _VIDEO_NONCRITICAL_EXCEPTIONS:
            pass
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        logging.debug("About to extract youtube info")
        info_dict = ydl.extract_info(video_url, download=False)
        logging.debug("Youtube info successfully extracted")
    return info_dict


def get_playlist_videos(playlist_url: str, *, use_cookies: bool = False, cookies: Optional[dict[str, Any] | str] = None):
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'quiet': True
    }
    if use_cookies and cookies:
        try:
            cookie_header = _cookies_to_header_value(cookies)
            if cookie_header:
                ydl_opts.setdefault('http_headers', {})['Cookie'] = cookie_header
        except _VIDEO_NONCRITICAL_EXCEPTIONS:
            pass

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

        if 'entries' in info:
            video_urls = [entry['url'] for entry in info['entries']]
            playlist_title = info['title']
            return video_urls, playlist_title
        else:
            print("No videos found in the playlist.")
            return [], None


def download_video(
    video_url,
    download_path,
    info_dict,
    download_video_flag,
    current_whisper_model=None,
    use_cookies: bool = False,
    cookies=None,
):
    """
    Download media using yt-dlp and return the path to the downloaded file.

    By default downloads the best available audio stream to minimise bandwidth.
    When ``download_video_flag`` is True, the best muxed audio+video stream is
    downloaded instead so the original media can be retained.
    """
    block_override: Optional[bool] = None
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING"):
        block_override = False
    policy_result = evaluate_url_policy(video_url, block_private_override=block_override)
    if not getattr(policy_result, "allowed", False):
        reason = policy_result.reason or "URL blocked by security policy"
        raise ValueError(f"URL blocked by security policy: {reason}")
    download_dir = Path(download_path)
    download_dir.mkdir(parents=True, exist_ok=True)

    # Reject when the declared size is already above the configured maximum.
    declared_size = _extract_declared_filesize(info_dict) if info_dict else None
    if (
        declared_size
        and DEFAULT_MAX_VIDEO_FILE_SIZE_BYTES
        and declared_size > DEFAULT_MAX_VIDEO_FILE_SIZE_BYTES
    ):
        raise ValueError(
            f"Declared download size {declared_size / (1024 * 1024):.2f}MB exceeds "
            f"maximum allowed {DEFAULT_MAX_VIDEO_FILE_SIZE_BYTES / (1024 * 1024):.2f}MB."
        )

    format_string = "bestvideo+bestaudio/best" if download_video_flag else "bestaudio/best"
    outtmpl = str(download_dir / "%(title).200B-%(id)s.%(ext)s")

    ydl_opts: dict[str, Any] = {
        "format": format_string,
        "restrictfilenames": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": False,
        "outtmpl": outtmpl,
        "paths": {"home": str(download_dir)},
        "retries": 3,
        "continuedl": True,
        "overwrites": False,
        "concurrent_fragment_downloads": 3,
        "postprocessors": [],
    }

    if not download_video_flag:
        # Convert to a stable audio container for downstream processing.
        ydl_opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
                "preferredquality": "192",
            }
        ]

    if use_cookies and cookies:
        cookie_header = _cookies_to_header_value(cookies)
        if cookie_header:
            ydl_opts.setdefault("http_headers", {})["Cookie"] = cookie_header

    downloaded_path: Optional[Path] = None

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(video_url, download=True)

            def _collect_candidate_paths(data: dict[str, Any]) -> list[Path]:
                candidates: list[Path] = []
                for key in ("filepath", "_filename", "filename"):
                    value = data.get(key)
                    if value:
                        candidates.append(Path(value))
                return candidates

            candidates: list[Path] = []

            requested = result.get("requested_downloads") or []
            for entry in requested:
                candidates.extend(_collect_candidate_paths(entry))

            # Some providers expose the final filepath directly on the result dict.
            candidates.extend(_collect_candidate_paths(result))

            # Fallback to the template yt-dlp would have used.
            if not candidates:
                candidates.append(Path(ydl.prepare_filename(result)))

        for candidate in candidates:
            if candidate.exists():
                downloaded_path = candidate
                break
            # Some postprocessors return relative paths; resolve relative to download dir.
            resolved_candidate = (download_dir / candidate.name).resolve()
            if resolved_candidate.exists():
                downloaded_path = resolved_candidate
                break

        if not downloaded_path:
            raise FileNotFoundError("Unable to determine downloaded file path.")

        logging.info(f"Downloaded media for transcription: {downloaded_path}")
        return str(downloaded_path)

    except _VIDEO_NONCRITICAL_EXCEPTIONS as exc:
        logging.exception(f"Failed to download media from {video_url}: {exc}")
        raise

def extract_video_info(url):
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)

            # Log only a subset of the info to avoid overwhelming the logs
            log_info = {
                'title': info.get('title'),
                'duration': info.get('duration'),
                'upload_date': info.get('upload_date')
            }
            logging.debug(f"Extracted info for {url}: {log_info}")

            return info
    except _VIDEO_NONCRITICAL_EXCEPTIONS as e:
        logging.exception(f"Error extracting video info for {url}: {str(e)}")
        return None


def get_youtube_playlist_urls(playlist_id):
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(f'https://www.youtube.com/playlist?list={playlist_id}', download=False)
        return [entry['url'] for entry in result['entries'] if entry.get('url')]


def parse_and_expand_urls(urls):
    logging.info(f"Starting parse_and_expand_urls with input: {urls}")
    expanded_urls = []

    for url in urls:
        try:
            logging.info(f"Processing URL: {url}")
            parsed_url = urlparse(url)
            logging.debug(f"Parsed URL components: {parsed_url}")

            # YouTube playlist handling
            netloc_lower = (parsed_url.netloc or "").lower()
            host_without_port = netloc_lower.split(":", 1)[0]
            query_params = parse_qs(parsed_url.query)
            youtube_playlist_hosts = (
                "youtube.com",
                "www.youtube.com",
                "m.youtube.com",
                "music.youtube.com",
                "youtube-nocookie.com",
            )

            def _matches_youtube_host(hostname: str, _hwp=host_without_port) -> bool:
                return (
                    _hwp == hostname
                    or _hwp.endswith(f".{hostname}")
                )

            if any(_matches_youtube_host(host) for host in youtube_playlist_hosts):
                playlist_ids = query_params.get("list")
                if playlist_ids:
                    playlist_id = playlist_ids[0]
                    logging.info(f"Detected YouTube playlist with ID: {playlist_id}")
                    playlist_urls = get_youtube_playlist_urls(playlist_id)
                    logging.info(f"Expanded playlist URLs: {playlist_urls}")
                    if playlist_urls:
                        expanded_urls.extend(playlist_urls)
                    else:
                        logging.warning(f"No entries found for playlist '{url}'. Keeping original URL.")
                        expanded_urls.append(url)
                    continue

            # YouTube short URL handling
            if 'youtu.be' in parsed_url.netloc:
                video_id = parsed_url.path.lstrip('/')
                full_url = f'https://www.youtube.com/watch?v={video_id}'
                logging.info(f"Expanded YouTube short URL to: {full_url}")
                expanded_urls.append(full_url)

            # Vimeo handling
            elif 'vimeo.com' in (parsed_url.netloc or parsed_url.path).lower():
                # Respect the supplied host while ensuring the URL is fully qualified.
                scheme = parsed_url.scheme or 'https'
                netloc = parsed_url.netloc
                path = parsed_url.path

                if not netloc:
                    # Handle scheme-less inputs like "vimeo.com/12345"
                    candidate_netloc, _, remainder = parsed_url.path.partition('/')
                    netloc = candidate_netloc
                    path = f'/{remainder}' if remainder else ''

                # Normalise empty paths to '/' so downstream clients get a valid URL.
                normalized_path = path or '/'

                # Upgrade to HTTPS for canonical hosts while preserving other subdomains verbatim.
                if netloc.lower() in {'vimeo.com', 'www.vimeo.com'}:
                    scheme = 'https'

                full_url = urlunparse((scheme, netloc, normalized_path, '', parsed_url.query, parsed_url.fragment))
                logging.info(f"Processed Vimeo URL: {full_url}")
                expanded_urls.append(full_url)

            # Add more platform-specific handling here

            else:
                logging.info(f"URL not recognized as special case, adding as-is: {url}")
                expanded_urls.append(url)

        except _VIDEO_NONCRITICAL_EXCEPTIONS as e:
            logging.exception(f"Error processing URL {url}: {str(e)}")
            expanded_urls.append(url)

    logging.info(f"Final expanded URLs: {expanded_urls}")
    return expanded_urls


HTTPONLY_PREFIX = '#HttpOnly_'
_HTTPONLY_PREFIX_LOWER = HTTPONLY_PREFIX.lower()


def _parse_netscape_cookie_export(text: str) -> list[str]:
    """Return cookie name=value pairs from a Netscape/Mozilla cookie export blob."""
    pairs: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower_line = line.lower()
        if lower_line.startswith(_HTTPONLY_PREFIX_LOWER):
            line = line[len(HTTPONLY_PREFIX):]
        elif line.startswith('#'):
            continue
        fields = line.split('\t')
        if len(fields) < 7:
            # Fallback to whitespace splitting if tabs are missing
            fields = [segment for segment in line.split(' ') if segment]
        if len(fields) < 7:
            continue
        name, value = fields[5], fields[6]
        if not name:
            continue
        pairs.append(f"{name}={value}")
    return pairs


def _cookies_to_header_value(cookies) -> Optional[str]:
    """Convert JSON string, dict, or Netscape export cookies into a Cookie header value."""
    try:
        if cookies is None:
            return None
        if isinstance(cookies, str):
            text = cookies.strip()
            if not text:
                return None
            try:
                cookies = json.loads(text)
            except json.JSONDecodeError:
                pairs = _parse_netscape_cookie_export(text)
                return "; ".join(pairs) if pairs else None
        if isinstance(cookies, dict):
            parts = []
            for k, v in cookies.items():
                if k is None or v is None:
                    continue
                k = str(k).strip()
                v = str(v).strip()
                if not k:
                    continue
                parts.append(f"{k}={v}")
            return "; ".join(parts) if parts else None
        return None
    except _VIDEO_NONCRITICAL_EXCEPTIONS:
        return None


def extract_metadata(url, use_cookies=False, cookies=None):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'skip_download': True,
    }

    if use_cookies and cookies:
        cookie_header = _cookies_to_header_value(cookies)
        if cookie_header:
            ydl_opts.setdefault('http_headers', {})['Cookie'] = cookie_header
        else:
            logging.warning("Invalid cookie input provided; continuing without cookies header.")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            metadata = {
                'title': info.get('title'),
                'uploader': info.get('uploader'),
                'upload_date': info.get('upload_date'),
                'view_count': info.get('view_count'),
                'like_count': info.get('like_count'),
                'duration': info.get('duration'),
                'tags': info.get('tags'),
                'description': info.get('description')
            }

            # Create a safe subset of metadata to log
            safe_metadata = {
                'title': metadata.get('title', 'No title'),
                'duration': metadata.get('duration', 'Unknown duration'),
                'upload_date': metadata.get('upload_date', 'Unknown upload date'),
                'uploader': metadata.get('uploader', 'Unknown uploader')
            }

            logging.info(f"Successfully extracted metadata for {url}: {safe_metadata}")
            return metadata
        except _VIDEO_NONCRITICAL_EXCEPTIONS as e:
            logging.exception(f"Error extracting metadata for {url}: {str(e)}")
            return None


def generate_timestamped_url(url, hours, minutes, seconds):
    # Extract video ID from the URL
    video_id_match = re.search(r'(?:v=|)([0-9A-Za-z_-]{11}).*', url)
    if not video_id_match:
        return "Invalid YouTube URL"

    video_id = video_id_match.group(1)

    # Calculate total seconds
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)

    # Generate the new URL
    new_url = f"https://www.youtube.com/watch?v={video_id}&t={total_seconds}s"

    return new_url


# New FastAPI ingestion functions
def process_videos(
    inputs: list[str],
    start_time: Optional[str],
    end_time: Optional[str],
    diarize: bool,
    vad_use: bool,
    transcription_model: str,
    transcription_language: Optional[str],
    perform_analysis: bool,
    custom_prompt: Optional[str],
    system_prompt: Optional[str],
    perform_chunking: bool,
    chunk_method: Optional[str],
    max_chunk_size: int,
    chunk_overlap: int,
    use_adaptive_chunking: bool,
    use_multi_level_chunking: bool,
    chunk_language: Optional[str],
    summarize_recursively: bool,
    api_name: Optional[str],
    # api_key removed - retrieved from server config
    use_cookies: bool,
    cookies: Optional[str],
    timestamp_option: bool,
    perform_confabulation_check: bool, # Renamed from confab_checkbox
    hotwords: Optional[Sequence[str] | str] = None,
    temp_dir: Optional[str] = None, # Added temp_dir argument
    keep_original: bool = False, # Add if needed for intermediate files
    perform_diarization:bool = False,
    user_id: Optional[int] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    """
    Processes multiple videos or local file paths, transcribes, summarizes,
    and optionally stores in the DB (if store_in_db=True).

    This function was adapted from your old `process_videos_with_error_handling()`
    but with Gradio references removed.

    :param inputs: A list of either URLs or local file paths.
    :param start_time: Start time for partial transcription (e.g. "1:30" or "90").
    :param end_time: End time for partial transcription.
    :param diarize: Enable speaker diarization.
    :param vad_use: Enable Voice Activity Detection.
    :param transcription_model: Name of the transcription model to use.
    :param transcription_language: Language for transcription.
    :param hotwords: Optional hotword hints (list/sequence or JSON/CSV string).
                     Primarily used by VibeVoice-ASR and ignored elsewhere.
    :param perform_analysis: If True, perform analysis on the transcript.
    :param custom_prompt: The user’s custom text prompt for summarization.
    :param system_prompt: The system prompt for the LLM.
    :param perform_chunking: If True, break transcripts into chunks before summarizing.
    :param chunk_method: "words", "sentences", etc.
    :param max_chunk_size: Maximum chunk size for chunking.
    :param chunk_overlap: Overlap size for chunking.
    :param use_adaptive_chunking: Whether to adapt chunk sizes by text complexity.
    :param use_multi_level_chunking: If True, chunk in multiple passes.
    :param chunk_language: The language for chunking logic.
    :param summarize_recursively: If True, do multi-pass summarization of chunk summaries.
    :param api_name: The LLM API name (e.g., "openai").
    # api_key parameter removed - API keys are retrieved from server config
    :param use_cookies: If True, use cookies for authenticated video downloads.
    :param cookies: The user-supplied cookies in JSON or Netscape format.
    :param timestamp_option: If True, keep timestamps in final transcript.
    :param perform_confabulation_check: If True, run confabulation check on the summary.
    :param keep_original: If True, keep the downloaded file
    :param perform_diarization: If True, perform diarization on inputs
    :param user_id: Identifier for the requesting user; used for downstream logging/context.
    :param cancel_check: Optional callable that returns True when processing should be cancelled.
    :return: A dict with the overall results, e.g.:
             {
               "processed_count": int,
               "errors_count": int,
               "errors": [...],
               "results": [...],
               "confabulation_results": "..."
             }
    """
    logging.info(f"Starting process_videos (DB-agnostic) for {len(inputs)} inputs.")
    expanded_inputs = parse_and_expand_urls(inputs)
    if expanded_inputs != inputs:
        logging.info(f"Expanded playlist and shortcut URLs into {len(expanded_inputs)} concrete entries.")
    inputs = expanded_inputs
    errors: list[str] = []
    warnings_accum: list[str] = []
    results = []

    def _is_cancelled() -> bool:
        """Return True if cancellation is requested, logging callback errors."""
        if cancel_check is None:
            return False
        try:
            return bool(cancel_check())
        except _VIDEO_NONCRITICAL_EXCEPTIONS as exc:
            logging.warning(f"cancel_check raised an error: {exc}")
            return False

    def _cancelled_result(input_ref: str, processing_source: Optional[str] = None) -> dict[str, Any]:
        """Build a standard cancelled result payload."""
        return {
            "status": "Cancelled",
            "input_ref": input_ref,
            "processing_source": processing_source or input_ref,
            "media_type": "video",
            "metadata": {},
            "content": None,
            "segments": None,
            "chunks": None,
            "analysis": None,
            "analysis_details": {},
            "error": "Cancelled by user",
            "warnings": [],
            "kept_video_path": None,
        }

    from tldw_Server_API.app.core.exceptions import TranscriptionCancelled
    all_transcripts_for_confab: dict[str, str] = {}
    all_summaries_for_confab: dict[str, str] = {}

    # If user typed no inputs, bail out
    if not inputs:
        logging.warning("No input provided to process_videos()")
        return {
            "processed_count": 0,
            "errors_count": 1,
            "warnings_count": 0,
            "errors": ["No inputs provided."],
            "warnings": [],
            "results": [],
            "confabulation_results": None,
        }

    # Convert user times to seconds
    try:
        start_seconds = convert_to_seconds(start_time) if start_time else 0
        end_seconds = convert_to_seconds(end_time) if end_time else None
    except ValueError as exc:
        error_msg = f"Invalid time parameter: {exc}"
        logging.error(error_msg)
        return {
            "processed_count": 0,
            "errors_count": len(inputs),
            "warnings_count": 0,
            "errors": [error_msg],
            "warnings": [],
            "results": [
                {
                    "status": "Error",
                    "input_ref": src,
                    "processing_source": src,
                    "media_type": "video",
                    "metadata": {},
                    "content": None,
                    "segments": None,
                    "chunks": None,
                    "analysis": None,
                    "analysis_details": {},
                    "error": error_msg,
                    "warnings": None,
                }
                for src in inputs
            ],
            "confabulation_results": None,
        }

    # Enforce temp_dir usage
    if not temp_dir:
        # If None is passed despite Fix #1, something is wrong upstream.
        logging.error("CRITICAL: process_videos called without a valid temp_dir path.")
        # Return an error immediately or raise, depending on desired behavior
        return {
            "processed_count": 0,
            "errors_count": len(inputs),
            "warnings_count": 0,
            "errors": ["Internal Error: Processing temporary directory was not provided."],
            "warnings": [],
            "results": [{"status": "Error", "input_ref": inp, "error": "Internal processing setup error"} for inp in inputs],
            "confabulation_results": None,
        }
    processing_temp_dir = Path(temp_dir)
    # Ensure the directory exists (it should, as TempDirManager creates it)
    if not processing_temp_dir.is_dir():
         logging.error(f"CRITICAL: Provided temp_dir '{processing_temp_dir}' does not exist or is not a directory.")
         # Handle error appropriately
         return {
             "processed_count": 0, "errors_count": len(inputs),
             "warnings_count": 0,
             "errors": [f"Internal Error: Invalid temporary directory '{processing_temp_dir}'."],
             "warnings": [],
             "results": [{"status": "Error", "input_ref": inp, "error": "Internal processing setup error"} for inp in inputs],
             "confabulation_results": None,
         }
    logging.info(f"process_videos using provided temporary directory: {processing_temp_dir}")

    for idx, video_input in enumerate(inputs):
        video_start_time = datetime.now()
        if _is_cancelled():
            logger.info("Cancellation detected before processing video input.")
            for remaining_input in inputs[idx:]:
                results.append(_cancelled_result(remaining_input))
            break
        try:
            # Pass necessary parameters down, including temp_dir
            single_result = process_single_video(
                video_input=video_input,
                start_seconds=start_seconds,
                end_seconds=end_seconds, # Pass end_seconds down
                diarize=diarize,
                vad_use=vad_use,
                transcription_model=transcription_model,
                transcription_language=transcription_language,
                hotwords=hotwords,
                perform_analysis=perform_analysis,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                perform_chunking=perform_chunking,
                chunk_method=chunk_method,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
                use_adaptive_chunking=use_adaptive_chunking,
                use_multi_level_chunking=use_multi_level_chunking,
                chunk_language=chunk_language,
                summarize_recursively=summarize_recursively,
                api_name=api_name,
                # api_key removed - retrieved from server config
                use_cookies=use_cookies,
                cookies=cookies,
                timestamp_option=timestamp_option,
                temp_dir=str(processing_temp_dir), # Pass temp dir path
                keep_intermediate_audio=False, # Pass if needed
                perform_diarization=perform_diarization,
                keep_original=keep_original,
                user_id=user_id,
                cancel_check=cancel_check,
            )

            results.append(single_result) # Append regardless of status
            if single_result.get("status") == "Cancelled":
                for remaining_input in inputs[idx + 1:]:
                    results.append(_cancelled_result(remaining_input))
                break

            if single_result.get("status") == "Success":
                item_warnings = single_result.get("warnings") or []
                if item_warnings:
                    warnings_accum.extend(item_warnings)
                # Record per-item success metric
                log_counter(
                    metric_name="videos_processed_total",
                    labels={"whisper_model": transcription_model, "api_name": (api_name or 'none')},
                    value=1,
                )

                # Prepare for potential confabulation check
                transcript_text = single_result.get("content", "") # Use 'transcript' key returned by single
                summary_text = single_result.get("analysis", "") # Use 'summary' key returned by single
                if transcript_text and summary_text:
                     all_transcripts_for_confab[video_input] = transcript_text
                     all_summaries_for_confab[video_input] = summary_text

                # Logging the timing
                video_end_time = datetime.now()
                processing_time = (video_end_time - video_start_time).total_seconds()
                log_histogram(
                    metric_name="video_processing_time_seconds",
                    value=processing_time,
                    labels={"whisper_model": transcription_model, "api_name": (api_name or "none")}
                )
            elif single_result.get("status") == "Error":
                # If status is "Error"
                if single_result.get("status") == "Error":
                    errors.append(single_result.get("error", "Unknown processing error"))
                item_warnings = single_result.get("warnings") or []
                if item_warnings:
                    warnings_accum.extend(item_warnings)

                # Log failure metric
                log_counter(
                    metric_name="videos_failed_total",
                    labels={"whisper_model": transcription_model, "api_name": (api_name or "none")},
                    value=1
                )
            elif single_result.get("status") == "Warning":
                # If status is "Warning"
                item_warnings = single_result.get("warnings") or []
                if item_warnings:
                    warnings_accum.extend(item_warnings)

        except TranscriptionCancelled:
            logger.info("Cancellation detected during video processing.")
            results.append(_cancelled_result(video_input))
            for remaining_input in inputs[idx + 1:]:
                results.append(_cancelled_result(remaining_input))
            break
        except _VIDEO_NONCRITICAL_EXCEPTIONS as exc:
            msg = f"Exception processing '{video_input}': {exc}"
            logging.exception(msg)
            public_error = "Video processing failed"
            errors.append(public_error)
            # Append an error result structure
            results.append({
                "status": "Error",
                "input_ref": video_input,
                "processing_source": video_input,
                "media_type": "video",
                "error": public_error,
                # Fill other fields with None/defaults
                "metadata": {}, "transcript": None, "segments": None, "chunks": None, "summary": None,
                "analysis_details": None, "warnings": None
            })

            # Log failure metric
            log_counter(
                metric_name="videos_failed_total",
                labels={"whisper_model": transcription_model, "api_name": (api_name or 'none')},
                value=1
            )

    # --- Recalculate counts based on the correctly populated 'results' list ---
    processed_count_calc = sum(1 for r in results if r.get("status") in {"Success", "Warning"})
    errors_count_calc = sum(1 for r in results if r.get("status") in {"Error", "Cancelled"})
    warnings_count_calc = sum(1 for r in results if r.get("status") == "Warning")

    # Optionally, run a confabulation check on the entire set of summaries
    confabulation_results = None
    if perform_confabulation_check:
        if not api_name:
            warning_msg = "Confabulation check requested, but no API name was provided; skipping g_eval."
            logging.warning(warning_msg)
            warnings_accum.append(warning_msg)
        elif not all_transcripts_for_confab:
            logging.info("Confabulation check requested, but no transcript/summary pairs were collected.")
            confabulation_results = "Confabulation check skipped: no transcript/summary pairs available."
        else:
            resolved_api_key = _resolve_eval_api_key(api_name)
            api_provider_key = api_name.lower().strip()
            provider_requires_key = (
                api_provider_key in _PROVIDERS_REQUIRING_KEYS
                or custom_openai_provider_number(api_provider_key) is not None
            )
            if provider_requires_key and not resolved_api_key:
                warning_msg = f"Confabulation check skipped: missing API key for provider '{api_name}'."
                logging.warning(warning_msg)
                confabulation_results = warning_msg
                warnings_accum.append(warning_msg)
            else:
                confab_results = []
                user_identifier = str(user_id) if user_id is not None else None
                for url, transcript in all_transcripts_for_confab.items():
                    summary_text = all_summaries_for_confab.get(url)
                    if not summary_text:
                        logging.warning(f"Confabulation check skipped for {url}: missing summary text.")
                        continue
                    try:
                        pair_result = run_geval(
                            transcript,
                            summary_text,
                            resolved_api_key,
                            api_name,
                            user_identifier=user_identifier,
                        )
                        confab_results.append(f"URL: {url} - {pair_result}")
                    except _VIDEO_NONCRITICAL_EXCEPTIONS as confab_err:
                        logging.exception(f"Confabulation check failed for {url}: {confab_err}")
                        confab_results.append(f"URL: {url} - Confabulation error: {confab_err}")

                if confab_results:
                    confabulation_results = "Confabulation checks completed:\n" + "\n".join(confab_results)
                else:
                    confabulation_results = "Confabulation check completed: no valid transcript/summary pairs to evaluate."

    logger.opt(lazy=True).debug(
        "process_videos DEBUG: Final results list before return: {}",
        lambda: json.dumps(results, indent=2, default=str),
    )
    logger.debug(f"process_videos DEBUG: Calculated processed_count: {processed_count_calc}")
    logger.debug(f"process_videos DEBUG: Calculated errors_count: {errors_count_calc}")
    logger.debug(f"process_videos DEBUG: Calculated warnings_count: {warnings_count_calc}")

    return {
        "processed_count": processed_count_calc,
        "errors_count": errors_count_calc,
        "warnings_count": warnings_count_calc,
        "errors": errors, # List of error messages
        "warnings": warnings_accum,
        "results": results,
        "confabulation_results": confabulation_results
    }


def process_single_video(
    video_input: str,
    start_seconds: int,
    end_seconds: Optional[int], # Ensure this is used if needed by transcription/summarization
    diarize: bool,
    vad_use: bool,
    transcription_model: str,
    transcription_language: Optional[str],
    perform_analysis: bool,
    custom_prompt: Optional[str],
    system_prompt: Optional[str],
    perform_chunking: bool,
    chunk_method: Optional[str],
    max_chunk_size: int,
    chunk_overlap: int,
    use_adaptive_chunking: bool,
    use_multi_level_chunking: bool,
    chunk_language: Optional[str],
    summarize_recursively: bool,
    api_name: Optional[str],
    # api_key removed - retrieved from server config
    use_cookies: bool,
    cookies: Optional[str],
    timestamp_option: bool,
    temp_dir: str, # Expect temp_dir path from caller (e.g., TempDirManager context)
    hotwords: Optional[Sequence[str] | str] = None,
    keep_intermediate_audio: bool = False, # Flag to keep the WAV file from transcription
    perform_diarization: bool = False, # Flag to perform diarization
    keep_original: bool = False,
    user_id: Optional[int] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    """
    Processes a single video/file: Extracts metadata, downloads if URL,
    transcribes, optionally summarizes.
    Returns a dict matching MediaItemProcessResponse structure.
    'input_ref' should hold the original URL/path passed in video_input.
    'processing_source' should hold the path of the file actually processed.
    cancel_check: Optional callable that returns True when processing should be cancelled.
    Hotwords are forwarded to STT providers that support them (for example
    VibeVoice-ASR) and ignored elsewhere.
    """
    # --- Initialize result with the ORIGINAL input reference ---
    processing_result = {
        "status": "Pending",
        "input_ref": video_input,  # Store the original URL or path here
        "processing_source": video_input, # Start with original, update if downloaded/copied
        "media_type": "video",
        "metadata": {},
        "content": "", # Corresponds to 'transcript'
        "segments": None,
        "chunks": None,
        "analysis": None, # Corresponds to 'summary'
        "analysis_details": {},
        "error": None,
        "warnings": [],
        "kept_video_path": None,
    }
    local_file_path_for_transcription = None
    # Temp dir for download is provided by the caller (`temp_dir`)

    def _is_cancelled() -> bool:
        """Return True if cancellation is requested, logging callback errors."""
        if cancel_check is None:
            return False
        try:
            return bool(cancel_check())
        except _VIDEO_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"cancel_check raised an error: {exc}")
            return False

    from tldw_Server_API.app.core.exceptions import TranscriptionCancelled

    try:
        logger.info(f"Processing single video input: {video_input}") # Log original
        if _is_cancelled():
            processing_result["status"] = "Cancelled"
            processing_result["error"] = "Cancelled by user"
            return processing_result

        # Check if it's a URL - handle cases without protocol
        parsed_url = urlparse(video_input)
        is_remote = parsed_url.scheme in ('http', 'https')

        # If no scheme but looks like a URL (starts with www. or contains common video domains)
        if not is_remote and (video_input.startswith('www.') or
                              any(domain in video_input.lower() for domain in ['youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com'])):
            video_input = f"https://{video_input}"
            parsed_url = urlparse(video_input)
            is_remote = True
            logger.info(f"Added https:// prefix to URL: {video_input}")
            # Update the processing result with the corrected URL
            processing_result["input_ref"] = video_input
            processing_result["processing_source"] = video_input

        processing_temp_dir = Path(temp_dir)

        # 1. Get Metadata & Determine LOCAL Processing Path
        if is_remote:
            block_override: Optional[bool] = None
            if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING"):
                block_override = False
            policy_result = evaluate_url_policy(video_input, block_private_override=block_override)
            if not getattr(policy_result, "allowed", False):
                reason = policy_result.reason or "URL blocked by security policy"
                err_msg = f"URL blocked by security policy: {reason}"
                processing_result.update({"status": "Error", "error": err_msg})
                return processing_result
            logger.info("Input is URL. Extracting metadata and downloading...")
            info_dict = extract_metadata(video_input, use_cookies, cookies)
            if not info_dict:
                raise ValueError(f"Failed to extract metadata for URL: {video_input}")
            processing_result["metadata"] = info_dict
            logger.debug(f"Metadata extracted for {video_input}")

            download_target_dir_str = str(processing_temp_dir)
            logger.info(f"Downloading URL to directory: {download_target_dir_str}")
            download_video_flag = bool(keep_original)
            if download_video_flag:
                logger.info("keep_original requested; downloading full video stream for archival.")
            try:
                downloaded_path = download_video(
                    video_url=video_input,
                    download_path=download_target_dir_str,
                    info_dict=info_dict,
                    download_video_flag=download_video_flag,
                    use_cookies=use_cookies,
                    cookies=cookies,
                )
            except NotImplementedError as exc:
                err_msg = str(exc) or "Video download is currently disabled."
                logging.error(err_msg)
                processing_result.update({"status": "Error", "error": err_msg})
                return processing_result

            if not downloaded_path or not os.path.exists(downloaded_path):
                raise FileNotFoundError(
                    f"Download failed or file not found (target in {download_target_dir_str}) for URL: {video_input}"
                )

            safe_downloaded = resolve_safe_local_path(
                Path(downloaded_path),
                processing_temp_dir,
            )
            if safe_downloaded is None:
                raise FileNotFoundError(
                    "Downloaded file path rejected outside temp directory."
                )
            _validate_downloaded_url_video_file(safe_downloaded)
            _enforce_download_quota(user_id, safe_downloaded)
            local_file_path_for_transcription = str(safe_downloaded)
            # *** Update only the processing_source, keep original input_ref ***
            processing_result["processing_source"] = local_file_path_for_transcription
            logger.info(f"Download successful. Using local path: {local_file_path_for_transcription}")

        else:
            # Input is already a local file path
            safe_local_path = resolve_safe_local_path(
                Path(video_input),
                processing_temp_dir,
            )
            if safe_local_path is None:
                raise FileNotFoundError(
                    f"Local file path rejected outside temp directory: {video_input}"
                )
            if not safe_local_path.exists():
                raise FileNotFoundError(f"Local file not found: {video_input}")
            local_file_path_for_transcription = str(safe_local_path)
            # *** Update only the processing_source, keep original input_ref ***
            processing_result["processing_source"] = local_file_path_for_transcription
            # Extract/create minimal metadata for local files if not already present
            if not processing_result.get("metadata"):
                 # Basic info; could potentially use ffprobe or similar for more details if needed
                 path_obj = safe_local_path
                 info_dict = {
                     "title": path_obj.stem,
                     "description": "Local file",
                     "webpage_url": f"local://{path_obj.name}",
                     "source_filename": path_obj.name,
                     # Add other fields as None or extract if possible
                 }
                 processing_result["metadata"] = info_dict
            logger.info(f"Input is local file: {local_file_path_for_transcription}")

        if keep_original and local_file_path_for_transcription:
            source_path = Path(local_file_path_for_transcription).resolve()
            if _VIDEO_STORAGE_ROOT in source_path.parents:
                processing_result["kept_video_path"] = str(source_path)
            else:
                kept_path = _store_video_file(source_path, user_id)
                if kept_path:
                    processing_result["kept_video_path"] = str(kept_path)
                else:
                    warn_msg = (
                        f"Unable to retain original video '{source_path.name}' due to storage limits or errors."
                    )
                    # Add warning only if storage failed and warn_msg defined
                    logging.warning(warn_msg)
                    processing_result["warnings"].append(warn_msg)

        # 2. Perform Transcription using the LOCAL file path
        logging.info(f"Calling perform_transcription with LOCAL path: {local_file_path_for_transcription}")
        # Ensure perform_transcription is correctly imported
        # Note: Pass the PROCESSING TEMP DIR to perform_transcription if it needs
        # a place to put its *own* intermediate files (like the WAV).
        # Check the signature of perform_transcription. Assuming it takes `temp_dir` now.
        if _is_cancelled():
            processing_result["status"] = "Cancelled"
            processing_result["error"] = "Cancelled by user"
            return processing_result
        intermediate_wav_path, segments = perform_transcription(
            video_path=local_file_path_for_transcription, # THE LOCAL PATH
            offset=start_seconds,
            end_seconds=end_seconds,
            transcription_model=transcription_model,
            vad_use=vad_use,
            diarize=diarize,
            overwrite=False, # Usually False for safety unless specifically needed
            transcription_language=transcription_language,
            hotwords=hotwords,
            temp_dir=str(processing_temp_dir), # Pass temp dir for its use
            cancel_check=cancel_check,
        )

        # Check transcription results carefully
        if segments is None:
            error_msg = "Transcription failed (returned None segments)."
            # Check if intermediate_wav_path holds error info (depends on perform_transcription impl.)
            if isinstance(intermediate_wav_path, dict) and 'error' in intermediate_wav_path:
                error_msg = f"Transcription failed: {intermediate_wav_path['error']}"
            elif isinstance(intermediate_wav_path, str) and "error" in intermediate_wav_path.lower():
                # Less ideal check if it returns error string in path var
                error_msg = f"Transcription failed: {intermediate_wav_path}"

            processing_result.update({"status": "Error", "error": error_msg})
            logger.error(error_msg + f" Input: {video_input}")
            return processing_result # Return early on transcription failure

        logger.info(f"Transcription successful for {local_file_path_for_transcription}")
        processing_result["segments"] = segments
        # Derive transcript text with a mock/test-friendly fallback: when segments are empty and the
        # first return value looks like plain text (not a file path), treat it as the transcript.
        try:
            if (not segments) and isinstance(intermediate_wav_path, str) and (not os.path.exists(intermediate_wav_path)) and intermediate_wav_path.strip():
                derived_text = intermediate_wav_path
            else:
                derived_text = extract_text_from_segments(segments, include_timestamps=timestamp_option)
        except _VIDEO_NONCRITICAL_EXCEPTIONS:
            derived_text = extract_text_from_segments(segments, include_timestamps=timestamp_option)
        processing_result["content"] = derived_text
        processing_result["analysis_details"]["whisper_model"] = transcription_model
        processing_result["analysis_details"]["transcription_language"] = transcription_language
        # Add other relevant details like diarize, vad_use if needed

        # Cleanup intermediate audio file created by transcription (if applicable)
        if not keep_intermediate_audio and intermediate_wav_path and os.path.exists(intermediate_wav_path):
             try:
                 os.remove(intermediate_wav_path)
                 logger.debug(f"Removed intermediate transcription audio file: {intermediate_wav_path}")
             except _VIDEO_NONCRITICAL_EXCEPTIONS as e:
                 warn_msg = f"Failed to remove intermediate audio file: {intermediate_wav_path} ({e})"
                 logging.warning(warn_msg)
                 processing_result["warnings"].append(warn_msg)

        # 3. Format Transcript (Content)
        # Possibly strip timestamps based on flag
        if not timestamp_option and isinstance(segments, list):
            logger.debug("Removing timestamps from segments.")
            for seg in segments:
                # Using .pop with default None avoids errors if keys are missing
                seg.pop("Time_Start", None)
                seg.pop("Time_End", None)
                seg.pop("start", None) # Check for alternative keys used by whisper
                seg.pop("end", None)

        # Prepare main 'content' string (reuse derived content if present)
        transcription_text = processing_result.get("content") or extract_text_from_segments(segments, include_timestamps=timestamp_option)
        if transcription_text == _TRANSCRIPTION_EXTRACTION_ERROR_SENTINEL:
            error_msg = "Transcription failed: unable to extract text from generated segments."
            logging.error(f"{error_msg} Input: {video_input}")
            processing_result["status"] = "Error"
            processing_result["error"] = error_msg
            processing_result["content"] = None
            return processing_result
        processing_result["content"] = transcription_text
        if not transcription_text:
             warn_msg = "Transcription resulted in empty text content."
             logging.warning(warn_msg)
             processing_result["warnings"].append(warn_msg)

        # 4. Analysis (Chunking & Summarization) if requested and content exists
        analysis_text = None
        if perform_analysis and api_name and api_name.lower() != "none" and transcription_text:
            processing_result["analysis_details"]["llm_api"] = api_name
            processing_result["analysis_details"]["custom_prompt"] = custom_prompt
            processing_result["analysis_details"]["system_prompt"] = system_prompt

            # Maybe add metadata context to the text before chunking/summarizing?
            # text_context = f"Title: {processing_result['metadata'].get('title', 'N/A')}\n" \
            #                f"Author: {processing_result['metadata'].get('uploader', 'N/A')}\n\n" \
            #                f"{transcription_text}"
            text_to_analyze = transcription_text # Start with transcript

            if perform_chunking:
                logger.info(f"Performing chunking for {local_file_path_for_transcription}")
                chunk_opts = {
                    'method': chunk_method or 'sentences', # Default if None
                    'max_size': max_chunk_size,
                    'overlap': chunk_overlap,
                    'adaptive': use_adaptive_chunking,
                    'multi_level': use_multi_level_chunking,
                    'language': chunk_language or transcription_language or 'en' # Sensible language default
                }
                # FIXME - validate chunk_opts
                processing_result["analysis_details"]["chunking_options"] = chunk_opts
                try:
                    chunked_texts_list = improved_chunking_process(text_to_analyze, chunk_opts)
                    processing_result["chunks"] = chunked_texts_list
                    if not chunked_texts_list:
                         warn_msg = "Chunking yielded no chunks. Analysis will use full text."
                         logging.warning(warn_msg)
                         processing_result["warnings"].append(warn_msg)
                         # Fallback: Summarize original text if chunking fails/is empty
                         analysis_text = analyze(api_name, text_to_analyze, custom_prompt, None, system_message=system_prompt)  # Pass None for api_key

                    else:
                         logger.info(f"Chunking successful, created {len(chunked_texts_list)} chunks.")
                         chunk_summaries = []
                         # Summarize each chunk
                         for i, chunk_block in enumerate(chunked_texts_list):
                             chunk_text = chunk_block.get("text")
                             if chunk_text:
                                 try:
                                     csum = analyze(api_name, chunk_text, custom_prompt, None, system_message=system_prompt)  # Pass None for api_key
                                     if csum:
                                         chunk_summaries.append(csum)
                                         # Optionally store chunk summary in chunk metadata if needed later
                                         chunk_block.setdefault("metadata", {})["summary"] = csum
                                 except _VIDEO_NONCRITICAL_EXCEPTIONS as chunk_summ_err:
                                      warn_msg = f"Summarization failed for chunk {i}: {chunk_summ_err}"
                                      logging.warning(warn_msg)
                                      processing_result["warnings"].append(warn_msg)
                                      chunk_block.setdefault("metadata", {})["summary_error"] = str(chunk_summ_err)


                         if chunk_summaries:
                             # Combine chunk summaries
                             if summarize_recursively and len(chunk_summaries) > 1:
                                 logger.info("Performing recursive summarization on chunk summaries.")
                                 combined_chunk_summaries = "\n\n---\n\n".join(chunk_summaries) # Use separator
                                 try:
                                     analysis_text = analyze(api_name, combined_chunk_summaries, custom_prompt or "Summarize the key points from the preceding text sections.", None, system_message=system_prompt)  # Pass None for api_key
                                 except _VIDEO_NONCRITICAL_EXCEPTIONS as rec_summ_err:
                                     warn_msg = f"Recursive summarization failed: {rec_summ_err}"
                                     logging.warning(warn_msg)
                                     processing_result["warnings"].append(warn_msg)
                                     analysis_text = combined_chunk_summaries # Fallback
                             else:
                                 analysis_text = "\n\n---\n\n".join(chunk_summaries) # Simple join
                         else:
                              warn_msg = "Analysis: Chunk summarization yielded no results."
                              logging.warning(warn_msg)
                              processing_result["warnings"].append(warn_msg)

                except _VIDEO_NONCRITICAL_EXCEPTIONS as chunk_err:
                    warn_msg = f"Chunking process failed: {chunk_err}. Analysis will use full text."
                    logging.opt(exception=True).warning(warn_msg)
                    processing_result["warnings"].append(warn_msg)
                    # Fallback: Summarize original text if chunking fails
                    try:
                        analysis_text = analyze(api_name, text_to_analyze, custom_prompt, None, system_message=system_prompt)  # Pass None for api_key
                    except _VIDEO_NONCRITICAL_EXCEPTIONS as summ_err:
                         warn_msg = f"Summarization failed after chunking error: {summ_err}"
                         logging.exception(warn_msg)
                         processing_result["warnings"].append(warn_msg)

            else: # No chunking requested
                 logger.info(f"Performing single-pass analysis for {local_file_path_for_transcription}")
                 try:
                     analysis_text = analyze(api_name, text_to_analyze, custom_prompt, None, system_message=system_prompt)  # Pass None for api_key
                 except _VIDEO_NONCRITICAL_EXCEPTIONS as summ_err:
                     warn_msg = f"Summarization failed: {summ_err}"
                     logging.exception(warn_msg)
                     processing_result["warnings"].append(warn_msg)

            processing_result["analysis"] = analysis_text # Store final analysis/summary
            if not analysis_text:
                 warn_msg = "Analysis was performed but resulted in empty content."
                 logging.warning(warn_msg)
                 processing_result["warnings"].append(warn_msg)
            else:
                 logger.info("Analysis completed.")

        elif not transcription_text:
            logging.warning("Analysis skipped because transcription text is empty.")
        elif not api_name or api_name.lower() == "none":
             logging.info("Analysis skipped because no API name was provided.")
        else:
             logging.info("Analysis skipped (not requested).")


        # 5. Final Status
        # If we reached here without erroring out earlier, it's at least a partial success.
        # Downgrade to Warning if any warnings were recorded.
        if processing_result["error"]: # Should have returned earlier if error was fatal
             processing_result["status"] = "Error"
        elif processing_result["warnings"]:
            processing_result["status"] = "Warning"
        else:
            processing_result["status"] = "Success"

        logger.info(f"Finished processing {video_input}. Final status: {processing_result['status']}")
        processing_result["input_ref"] = video_input
        return processing_result

    except TranscriptionCancelled:
        processing_result["status"] = "Cancelled"
        processing_result["error"] = "Cancelled by user"
        processing_result["input_ref"] = video_input
        return processing_result
    except FileNotFoundError as e:
        logger.exception(f"File not found error processing {video_input}: {e}")
        processing_result["status"] = "Error"
        processing_result["error"] = str(e)
        # *** Ensure input_ref is original on error ***
        processing_result["input_ref"] = video_input
        return processing_result
    except ValueError as e: # Catch metadata or other value errors
        logger.exception(f"Value error processing {video_input}: {e}")
        processing_result["status"] = "Error"
        processing_result["error"] = str(e)
        # *** Ensure input_ref is original on error ***
        processing_result["input_ref"] = video_input
        return processing_result
    except _VIDEO_NONCRITICAL_EXCEPTIONS as e:
        # Catch-all for unexpected errors during the process
        logger.exception(f"Unexpected exception processing {video_input}: {e}")
        processing_result["status"] = "Error"
        processing_result["error"] = "Video processing failed"
        # *** Ensure input_ref is original on error ***
        processing_result["input_ref"] = video_input
        return processing_result

#
# End of Video_DL_Ingestion_Lib.py
#######################################################################################################################
