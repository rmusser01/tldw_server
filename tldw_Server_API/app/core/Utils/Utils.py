# Utils.py
from __future__ import annotations

import contextlib
import hashlib
import json
import mimetypes
import os
import re
import tempfile
import time
import unicodedata
import uuid
import zipfile
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import AnyStr
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

#########################################
# General Utilities Library
# This library is used to hold random utilities used by various other libraries.
#
####
####################
# Function Categories
#
#     Config loading
#     Misc-Functions
#     File-saving Function Definitions
#     UUID-Functions
#     Sanitization/Verification Functions
#     DB Config Loading
#     File Handling Functions
#
####################
# Function List
#
# 1. extract_text_from_segments(segments: List[Dict]) -> str
# 2. download_file(url, dest_path, expected_checksum=None, max_retries=3, delay=5)
# 3. verify_checksum(file_path, expected_checksum)
# 4. create_download_directory(title)
# 5. sanitize_filename(filename)
# 6. normalize_title(title)
# 7.
#
####################
#
# Import necessary libraries
import chardet
from loguru import logger

#
# 3rd-Party Imports
from tldw_Server_API.app.core.http_client import RetryPolicy, download, fetch

#
#######################################################################################################################
#
# Function Definitions

logging = logger

def extract_text_from_segments(segments, include_timestamps=True):
    """Extract transcript text from nested segment structures without logging raw content."""
    def _describe_segment_shape(data):
        """Summarize segment container shape for redacted diagnostic logs."""
        if isinstance(data, list):
            return f"list length={len(data)}"
        if isinstance(data, dict):
            return f"dict keys={len(data)}"
        return type(data).__name__

    logger.trace(f"Segments received: type={type(segments).__name__}, shape={_describe_segment_shape(segments)}")
    logger.trace(f"Type of segments: {type(segments)}")

    def _format_seconds_value(value):
        """Format numeric timestamp values without unnecessary trailing zeros."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if number == 0:
            number = 0.0
        return f"{number:.2f}".rstrip("0").rstrip(".")

    def extract_text_recursive(data):
        """Walk nested segment data and collect transcript text entries."""
        results = []
        if isinstance(data, dict):
            if 'Text' in data and isinstance(data['Text'], str):
                text_item = data['Text']
                if include_timestamps and 'Time_Start' in data and 'Time_End' in data:
                    start_ts = _format_seconds_value(data['Time_Start'])
                    end_ts = _format_seconds_value(data['Time_End'])
                    text_item = f"{start_ts}s - {end_ts}s | {text_item}"
                results.append(text_item)
            for key, value in data.items():
                if key == 'Text':
                    continue
                if isinstance(value, (dict, list)):
                    results.extend(extract_text_recursive(value))
        elif isinstance(data, list):
            for item in data:
                results.extend(extract_text_recursive(item))
        return results

    pieces = [piece.strip() for piece in extract_text_recursive(segments) if piece]

    if pieces:
        return '\n'.join(pieces)

    logger.error(
        f"Unable to extract text from segments: type={type(segments).__name__}, "
        f"shape={_describe_segment_shape(segments)}"
    )
    return "Error: Unable to extract transcription"

#
#
#######################
# Temp file cleanup
#
# Global list to keep track of downloaded files
downloaded_files = []

def cleanup_downloads():
    """Function to clean up downloaded files when the server exits."""
    for file_path in downloaded_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Cleaned up file: {file_path}")
        except OSError as e:
            logging.exception(f"Error cleaning up file {file_path}: {e}")

#
#
#######################################################################################################################


#######################################################################################################################
# Config loading
#



def get_project_root() -> str:
    """Return the absolute path to the repository root directory.

    Prefer a VCS marker (``.git``), then fall back to build metadata such as
    ``pyproject.toml`` alongside the top-level ``tldw_Server_API`` package.
    Retains the legacy fallback for environments where the project is bundled
    without those markers (e.g., certain tests).
    """
    current_path = Path(__file__).resolve()

    for candidate in current_path.parents:
        if (candidate / ".git").exists():
            project_root = str(candidate)
            logging.trace(f"Project root (.git sentinel): {project_root}")
            return project_root
        if (candidate / "pyproject.toml").exists() and (candidate / "tldw_Server_API").is_dir():
            project_root = str(candidate)
            logging.trace(f"Project root (pyproject sentinel): {project_root}")
            return project_root

    # Fallback: ensure we don't IndexError if the structure changes drastically.
    try:
        fallback_root = str(current_path.parents[4])
    except IndexError:
        try:
            fallback_root = str(current_path.parents[3])
        except IndexError:
            fallback_root = str(current_path.parent)
    logging.trace(f"Project root fallback: {fallback_root}")
    return fallback_root


def get_database_dir():
    """Get the absolute path to the database directory."""
    db_dir = os.path.join(get_project_root(), 'Databases')
    os.makedirs(db_dir, exist_ok=True)
    logging.trace(f"Database directory: {db_dir}")
    return db_dir


def get_database_path(db_name: str) -> str:
    """
    Get the full absolute path for a database file.
    Ensures the path is always within the Databases directory.
    """
    # Remove any directory traversal attempts
    safe_db_name = os.path.basename(db_name)
    path = os.path.join(get_database_dir(), safe_db_name)
    logging.trace(f"Database path for {safe_db_name}: {path}")
    return path


def get_project_relative_path(relative_path: str | os.PathLike[AnyStr]) -> str:
    """Convert a relative path to a path relative to the project root."""
    path = os.path.join(get_project_root(), str(relative_path))
    logging.trace(f"Project relative path for {relative_path}: {path}")
    return path

def get_chromadb_path():
    """Return the project-local ChromaDB storage path."""
    path = os.path.join(get_project_root(), 'Databases', 'chroma_db')
    logging.trace(f"ChromaDB path: {path}")
    return path

def ensure_directory_exists(path):
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)

# FIXME - update to include prompt path in return statement
# FIXME - remove api Key checks from config file and instead check .env file

global_api_endpoints = ["anthropic", "cohere", "google", "groq", "openai", "huggingface", "openrouter", "deepseek", "mistral", "custom_openai_api", "custom_openai_api_2", "llama", "ollama", "ooba", "kobold", "tabby", "vllm", "aphrodite"]

global_search_engines = [
    "baidu",
    "bing",
    "brave",
    "duckduckgo",
    "exa",
    "firecrawl",
    "google",
    "kagi",
    "searx",
    "serper",
    "tavily",
    "yandex",
    "sogou",
    "startpage",
    "stract",
]

openai_tts_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]





def format_api_name(api):
    """Return a display name for a configured API provider key."""
    name_mapping = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "cohere": "Cohere",
        "google": "Google",
        "groq": "Groq",
        "huggingface": "HuggingFace",
        "openrouter": "OpenRouter",
        "deepseek": "DeepSeek",
        "mistral": "Mistral",
        "custom_openai_api": "Custom-OpenAI-API",
        "custom_openai_api_2": "Custom-OpenAI-API-2",
        "llama": "Llama.cpp",
        "ooba": "Ooba",
        "kobold": "Kobold",
        "tabby": "TabbyAPI",
        "vllm": "VLLM",
        "ollama": "Ollama",
        "aphrodite": "Aphrodite"
    }
    return name_mapping.get(api, api.title())

#
# End of Config loading
#######################################################################################################################


#######################################################################################################################
#
# Misc-Functions

# Log file
# logging.basicConfig(filename='debug-runtime.log', encoding='utf-8', level=logging.DEBUG)

def format_metadata_as_text(metadata):
    """Format media metadata mappings into a human-readable multi-line summary."""
    if not metadata:
        return "No metadata available"

    formatted_text = "Video Metadata:\n"
    for key, value in metadata.items():
        if value is not None:
            if isinstance(value, list):
                # Join list items with commas
                formatted_value = ", ".join(str(item) for item in value)
            elif key == 'upload_date' and len(str(value)) == 8:
                # Format date as YYYY-MM-DD
                formatted_value = f"{value[:4]}-{value[4:6]}-{value[6:]}"
            elif key in ['view_count', 'like_count']:
                # Format large numbers with commas
                formatted_value = f"{value:,}"
            elif key == 'duration':
                # Convert seconds or time strings to HH:MM:SS format
                try:
                    total_seconds = convert_to_seconds(value)
                    hours, remainder = divmod(int(total_seconds), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    formatted_value = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                except (TypeError, ValueError):
                    formatted_value = str(value)
            else:
                formatted_value = str(value)

            # Replace underscores with spaces in the key name
            formatted_key = key.replace('_', ' ').capitalize()
            formatted_text += f"{formatted_key}: {formatted_value}\n"
    return formatted_text.strip()

# # Example usage:
# example_metadata = {
#     'title': 'Sample Video Title',
#     'uploader': 'Channel Name',
#     'upload_date': '20230615',
#     'view_count': 1000000,
#     'like_count': 50000,
#     'duration': 3725,  # 1 hour, 2 minutes, 5 seconds
#     'tags': ['tag1', 'tag2', 'tag3'],
#     'description': 'This is a sample video description.'
# }
#
# print(format_metadata_as_text(example_metadata))


def convert_to_seconds(time_str):
    """Convert seconds or HH:MM:SS-like duration values to whole seconds."""
    if not time_str:
        return 0

    time_str = str(time_str).strip()
    if not time_str:
        return 0

    def _as_int(value: float) -> int:
        """Round a non-negative duration component to an integer second."""
        if value < 0:
            raise ValueError("Time values must be non-negative")
        quantized = Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return int(quantized)

    # If it's already a plain number (allowing decimals), treat as seconds.
    try:
        if ":" not in time_str:
            return _as_int(float(time_str))
    except ValueError as exc:
        raise ValueError(f"Invalid time value '{time_str}'") from exc

    # Parse time string in format HH:MM:SS(.sss), MM:SS(.sss), or SS(.sss)
    time_parts = time_str.split(':')
    if len(time_parts) == 3:
        hours, minutes, seconds = time_parts
        try:
            total_seconds = (
                int(hours) * 3600 +
                int(minutes) * 60 +
                float(seconds)
            )
        except ValueError as exc:
            raise ValueError(f"Invalid HH:MM:SS value '{time_str}'") from exc
        return _as_int(total_seconds)
    if len(time_parts) == 2:
        minutes, seconds = time_parts
        try:
            total_seconds = int(minutes) * 60 + float(seconds)
        except ValueError as exc:
            raise ValueError(f"Invalid MM:SS value '{time_str}'") from exc
        return _as_int(total_seconds)
    if len(time_parts) == 1:
        try:
            return _as_int(float(time_parts[0]))
        except ValueError as exc:
            raise ValueError(f"Invalid seconds value '{time_str}'") from exc

    raise ValueError(f"Invalid time format '{time_str}'")


def truncate_content(content: str | None, max_length: int = 200) -> str | None:
    """Truncate content to the specified maximum length with ellipsis."""
    if not content:
        return content

    if len(content) <= max_length:
        return content

    return content[:max_length - 3] + "..."

#
# End of Misc-Functions
#######################################################################################################################


#######################################################################################################################
#
# File-saving Function Definitions
def save_to_file(video_urls, filename):
    """Write a list of video URLs to a newline-delimited text file."""
    with open(filename, 'w') as file:
        file.write('\n'.join(video_urls))
    logging.info(f"Video URLs saved to {filename}")


def save_segments_to_json(segments, file_name="transcription_segments.json"):
    """
    Save transcription segments to a JSON file.

    Parameters:
    segments (list): List of transcription segments
    file_name (str): Name of the JSON file to save (default: "transcription_segments.json")

    Returns:
    str: Path to the saved JSON file
    """
    # Ensure the Results directory exists
    os.makedirs("Results", exist_ok=True)

    # Full path for the JSON file
    json_file_path = os.path.join("Results", file_name)

    # Save segments to JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(segments, json_file, ensure_ascii=False, indent=4)

    return json_file_path


def safe_download(url: str, tmp_dir: Path, ext: str) -> Path:
    """
    Wrapper around download_file() that:
      1) builds a random filename inside tmp_dir
      2) returns the Path on success
    """
    dst = tmp_dir / (f"{uuid.uuid4().hex}{ext}")
    # checksum=None, max_retries=3, delay=5 keep the defaults
    download_file(url, str(dst))          # raises on failure
    return dst

def smart_download(url: str, tmp_dir: Path) -> Path:
    """
    • Chooses a filename & extension automatically
    • Calls download_file(url, dest_path)
    • Returns Path to downloaded file

    Order of extension preference:
      1. The URL path (e.g. “.md”, “.rst”, “.txt” …)
      2. The HTTP Content-Type header
      3. Fallback: “.bin”
    """
    # ---------- 1) try URL  -------------------------------------------------
    parsed = urlparse(url)
    guessed_ext = Path(parsed.path).suffix.lower()

    # ---------- 2) if no ext, probe with GET Range (prefer over HEAD)  ------
    if not guessed_ext:
        try:
            head = fetch(method="GET", url=url, allow_redirects=True, timeout=10, headers={"Range": "bytes=0-0"})
            ctype = head.headers.get("content-type", "")
            guessed_ext = mimetypes.guess_extension(ctype.split(";")[0].strip()) or ""
        except (OSError, RuntimeError, TypeError, ValueError):
            guessed_ext = ""

    # ---------- 3) final fallback  ------------------------------------------
    if not guessed_ext:
        guessed_ext = ".bin"

    # ---------- 4) build dest path  -----------------------------------------
    dest = tmp_dir / f"{uuid.uuid4().hex}{guessed_ext}"

    # ---------- 5) download  ------------------------------------------------
    download_file(url, str(dest))          # inherits retries / resume
    return dest


def download_file(url, dest_path, expected_checksum=None, max_retries=3, delay=5):
    """Download a URL to disk with retries, resume support, and optional checksum validation."""
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    # Use centralized downloader with resume+retries and atomic rename
    try:
        policy = RetryPolicy(attempts=max_retries, backoff_base_ms=int(delay * 1000))
        download(url=url, dest=dest_path, resume=True, retry=policy)
        if expected_checksum and not verify_checksum(dest_path, expected_checksum):
            try:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
            except OSError as cleanup_error:
                logging.exception(f"Failed to remove file after checksum mismatch: {cleanup_error}")
            raise ValueError("Downloaded file's checksum does not match the expected checksum")
        logging.info("Download complete and verified!")
        return dest_path
    except Exception as e:
        logging.exception(f"Download failed: {e}")
        raise

def download_file_if_missing(url: str, local_path: str) -> None:
    """
    Download a file from a URL if it does not exist locally.
    """
    if os.path.exists(local_path):
        logging.debug(f"File already exists locally: {local_path}")
        return
    logging.info(f"Downloading from {url} to {local_path}")
    dirpath = os.path.dirname(local_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    try:
        download(url=url, dest=local_path, resume=False)
    except Exception as e:
        logging.exception(f"Download failed: {e}")
        raise

def create_download_directory(title):
    """Create and return a sanitized Results subdirectory for downloaded media."""
    base_dir = "Results"
    # Remove characters that are illegal in Windows filenames and normalize
    safe_title = normalize_title(title, preserve_spaces=False)
    logging.debug(f"{title} successfully normalized")
    session_path = os.path.join(base_dir, safe_title)
    if not os.path.exists(session_path):
        os.makedirs(session_path, exist_ok=True)
        logging.debug(f"Created directory for downloaded video: {session_path}")
    else:
        logging.debug(f"Directory already exists for downloaded video: {session_path}")
    return session_path


def safe_read_file(file_path):
    """Read a text file using detected or fallback encodings, returning an error string on failure."""
    encodings = ['utf-8', 'utf-16', 'ascii', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']

    logging.info(f"Attempting to read file: {file_path}")

    try:
        with open(file_path, 'rb') as file:
            logging.debug(f"Reading file in binary mode: {file_path}")
            raw_data = file.read()
    except FileNotFoundError:
        logging.exception(f"File not found: {file_path}")
        return f"File not found: {file_path}"
    except (OSError, TypeError, ValueError) as e:
        logging.exception(f"An error occurred while reading the file: {e}")
        return f"An error occurred while reading the file: {e}"

    if not raw_data:
        logging.warning(f"File is empty: {file_path}")
        return ""

    # Use chardet to detect the encoding
    detected = chardet.detect(raw_data)
    if detected['encoding'] is not None:
        encodings.insert(0, detected['encoding'])
        logging.info(f"Detected encoding: {detected['encoding']}")

    for encoding in encodings:
        logging.info(f"Trying encoding: {encoding}")
        try:
            decoded_content = raw_data.decode(encoding)
            # Check if the content is mostly printable
            if not decoded_content:
                logging.info(f"Decoded content empty with encoding {encoding}, trying next")
                continue
            if sum(c.isprintable() for c in decoded_content) / len(decoded_content) > 0.90:
                logging.info(f"Successfully decoded file with encoding: {encoding}")
                return decoded_content
        except UnicodeDecodeError:
            logging.debug(f"Failed to decode with {encoding}")
            continue

    # If all decoding attempts fail, return the error message
    logging.error(f"Unable to decode the file {file_path}")
    return f"Unable to decode the file {file_path}"


#
# End of Files-saving Function Definitions
#######################################################################################################################


#######################################################################################################################
#
# UUID-Functions

def generate_unique_filename(base_path, base_filename):
    """Generate a unique filename by appending a counter if necessary."""
    filename = base_filename
    counter = 1
    while os.path.exists(os.path.join(base_path, filename)):
        name, ext = os.path.splitext(base_filename)
        filename = f"{name}_{counter}{ext}"
        counter += 1
    return filename


def generate_unique_identifier(file_path):
    """Build a local identifier from file timestamp, content hash, and filename."""
    filename = os.path.basename(file_path)
    timestamp = int(time.time())

    # Generate a hash of the file content
    hasher = hashlib.md5(usedforsecurity=False)
    with open(file_path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(block)
    content_hash = hasher.hexdigest()[:8]  # Use first 8 characters of the hash

    return f"local:{timestamp}:{content_hash}:{filename}"

#
# End of UUID-Functions
#######################################################################################################################


#######################################################################################################################
#
# Sanitization/Verification Functions

# Helper function to validate URL format
def is_valid_url(url: str) -> bool:
    """Return whether a string matches the accepted URL pattern."""
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None


def verify_checksum(file_path, expected_checksum):
    """Return whether a file's SHA-256 digest matches the expected checksum."""
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b''):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_checksum


def normalize_title(title, preserve_spaces=False):
    """Normalize a title into a filesystem-safe ASCII name."""
    # Normalize the string to 'NFKD' form and encode to 'ascii' ignoring non-ascii characters
    title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')

    if preserve_spaces:
        # Replace special characters with underscores, but keep spaces
        title = re.sub(r'[^\w\s\-.]', '_', title)
    else:
        # Replace special characters and spaces with underscores
        title = re.sub(r'[^\w\-.]', '_', title)

    # Replace multiple consecutive underscores with a single underscore
    title = re.sub(r'_+', '_', title)

    # Replace specific characters with underscores
    title = title.replace('/', '_').replace('\\', '_').replace(':', '_').replace('"', '_').replace('*', '_').replace(
        '?', '_').replace(
        '<', '_').replace('>', '_').replace('|', '_')

    return title.strip('_')


def clean_youtube_url(url):
    """Remove playlist query parameters from a YouTube URL."""
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if 'list' in query_params:
        query_params.pop('list')
    cleaned_query = urlencode(query_params, doseq=True)
    cleaned_url = urlunparse(parsed_url._replace(query=cleaned_query))
    return cleaned_url

def sanitize_filename(filename, *, max_total_length: int | None = None, extension: str | None = None):
    """
    Sanitizes a user-provided filename component.

    Behavior:
    - Removes forbidden characters entirely
    - Collapses whitespace and repeated dashes
    - Optionally enforces a total length cap (base + extension) preserving the extension

    Args:
        filename: The filename or base name to sanitize (callers often pass a stem w/o extension).
        max_total_length: If provided, ensures (sanitized_base + extension) length <= this cap.
        extension: Optional extension (including leading dot), used when enforcing the cap.

    Returns:
        A sanitized (and possibly truncated) filename or base component.
    """
    # 1) Remove forbidden characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', str(filename))
    # 2) Replace runs of whitespace with a single space
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    # 3) Replace consecutive dashes with a single dash
    sanitized = re.sub(r'-{2,}', '-', sanitized)

    if sanitized in {"", ".", ".."}:
        sanitized = "untitled"

    # Optional capping: preserve extension if provided and cap overall length conservatively
    if max_total_length is not None and max_total_length > 0:
        ext = extension or ''
        # Ensure extension starts with a dot if it looks like one; otherwise treat as raw suffix
        if ext and not ext.startswith('.') and len(ext) <= 6:  # small guard; callers should pass with dot
            ext = f'.{ext}'
        reserved = len(ext)
        available = max_total_length - reserved
        if available < 1:
            # Degenerate case: if cap is smaller than extension, drop extension consideration
            available = max_total_length
            ext = ''
        if len(sanitized) > available:
            sanitized = sanitized[:available]

    return sanitized


def format_transcription(content):
    """Convert escaped transcript newlines into simple HTML line breaks."""
    # Replace '\n' with actual line breaks
    content = content.replace('\\n', '\n')
    # Split the content by newlines first
    lines = content.split('\n')
    formatted_lines = []
    for line in lines:
        # Add extra space after periods for better readability
        line = line.replace('.', '. ').replace('.  ', '. ')

        # Split into sentences using a more comprehensive regex
        sentences = re.split('(?<=[.!?]) +', line)

        # Trim whitespace from each sentence and add a line break
        formatted_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        # Join the formatted sentences
        formatted_lines.append(' '.join(formatted_sentences))

    # Join the lines with HTML line breaks
    formatted_content = '<br>'.join(formatted_lines)

    return formatted_content

def sanitize_user_input(message):
    """
    Removes or escapes '{{' and '}}' to prevent placeholder injection.

    Args:
        message (str): The user's message.

    Returns:
        str: Sanitized message.
    """
    # Replace '{{' and '}}' with their escaped versions
    message = re.sub(r'\{\{', '{ {', message)
    message = re.sub(r'\}\}', '} }', message)
    return message

def format_file_path(file_path, fallback_path=None):
    """Return an existing file path, falling back to an alternate path when available."""
    if file_path and os.path.exists(file_path):
        logging.debug(f"File exists: {file_path}")
        return file_path
    elif fallback_path and os.path.exists(fallback_path):
        logging.debug(f"File does not exist: {file_path}. Returning fallback path: {fallback_path}")
        return fallback_path
    else:
        logging.debug(f"File does not exist: {file_path}. No fallback path available.")
        return None

#
# End of Sanitization/Verification Functions
#######################################################################################################################


#######################################################################################################################
#
# DB Config Loading


def get_db_config():
    """DEPRECATED: Use tldw_Server_API.app.core.DB_Management.DB_Manager.get_db_config instead.

    This thin wrapper forwards to the canonical DB manager to avoid drift.
    """
    logging.warning(
        "Utils.get_db_config() is deprecated; use DB_Management.DB_Manager.get_db_config() instead."
    )
    try:
        from tldw_Server_API.app.core.DB_Management.DB_Manager import get_db_config as _dbm_get
        return _dbm_get()
    except (AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError) as e:
        logging.exception(f"Failed to delegate to DB_Manager.get_db_config: {e}")
        # Preserve a minimal, safe default
        try:
            from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
            default_sqlite = str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id()))
        except (AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logging.exception(f"Failed to resolve default SQLite path via DatabasePaths: {exc}")
            raise
        return {
            'type': 'sqlite',
            'sqlite_path': default_sqlite,
            'elasticsearch_host': 'localhost',
            'elasticsearch_port': 9200,
        }

#
# End of DB Config Loading
#######################################################################################################################


#######################################################################################################################
#
# File Handling Functions

# Track temp files for cleanup
temp_files = []

def save_temp_file(file):
    """Persist an uploaded file-like object to a unique path in the system temp directory."""
    global temp_files
    temp_dir = tempfile.gettempdir()

    original_name = getattr(file, "name", "") or ""
    safe_name = os.path.basename(original_name)
    stem, ext = os.path.splitext(safe_name)
    if not stem:
        stem = "upload"
    unique_name = f"{stem}_{uuid.uuid4().hex}{ext}"

    temp_path = os.path.join(temp_dir, unique_name)
    if hasattr(file, "seek"):
        with contextlib.suppress(OSError, RuntimeError, ValueError):
            file.seek(0)
    data = file.read()
    if isinstance(data, str):
        data = data.encode('utf-8')
    with open(temp_path, 'wb') as f:
        f.write(data)
    if hasattr(file, "seek"):
        with contextlib.suppress(OSError, RuntimeError, ValueError):
            file.seek(0)
    temp_files.append(temp_path)
    return temp_path

def cleanup_temp_files():
    """Delete temporary files recorded by save_temp_file."""
    global temp_files
    for file_path in temp_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Removed temporary file: {file_path}")
            except OSError as e:
                logging.exception(f"Failed to remove temporary file {file_path}: {e}")
    temp_files.clear()

def generate_unique_id():
    """Return a unique uploaded-file identifier."""
    return f"uploaded_file_{uuid.uuid4()}"

class FileProcessor:
    """Handles file reading and name processing"""

    VALID_EXTENSIONS = {'.md', '.txt', '.zip'}
    ENCODINGS_TO_TRY = [
        'utf-8',
        'utf-16',
        'windows-1252',
        'iso-8859-1',
        'ascii'
    ]

    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """Detect the file encoding using chardet"""
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'

    @staticmethod
    def read_file_content(file_path: str) -> str:
        """Read file content with automatic encoding detection"""
        detected_encoding = FileProcessor.detect_encoding(file_path)

        # Try detected encoding first
        try:
            with open(file_path, encoding=detected_encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # If detected encoding fails, try others
            for encoding in FileProcessor.ENCODINGS_TO_TRY:
                try:
                    with open(file_path, encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue

            # If all encodings fail, use utf-8 with error handling
            with open(file_path, encoding='utf-8', errors='replace') as f:
                return f.read()

    @staticmethod
    def process_filename_to_title(filename: str) -> str:
        """Convert filename to a readable title"""
        # Remove extension
        name = os.path.splitext(filename)[0]

        # Look for date patterns
        date_pattern = r'(\d{4}[-_]?\d{2}[-_]?\d{2})'
        date_match = re.search(date_pattern, name)
        date_str = ""
        if date_match:
            try:
                date = datetime.strptime(date_match.group(1).replace('_', '-'), '%Y-%m-%d')
                date_str = date.strftime("%b %d, %Y")
                name = name.replace(date_match.group(1), '').strip('-_')
            except ValueError:
                pass

        # Replace separators with spaces
        name = re.sub(r'[-_]+', ' ', name)

        # Remove redundant spaces
        name = re.sub(r'\s+', ' ', name).strip()

        # Capitalize words, excluding certain words
        exclude_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        words = name.split()
        capitalized = []
        for i, word in enumerate(words):
            if i == 0 or word not in exclude_words:
                capitalized.append(word.capitalize())
            else:
                capitalized.append(word.lower())
        name = ' '.join(capitalized)

        # Add date if found
        if date_str:
            name = f"{name} - {date_str}"

        return name


class ZipValidator:
    """Validates zip file contents and structure"""

    MAX_ZIP_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_FILES = 100
    VALID_EXTENSIONS = {'.md', '.txt'}

    @staticmethod
    def _is_malicious_zip_path(filename: str) -> bool:
        """Return whether a zip entry path attempts absolute or parent traversal."""
        normalized = filename.replace("\\", "/")
        if normalized.startswith("/") or normalized.startswith("//"):
            return True
        if re.match(r"^[A-Za-z]:", normalized):
            return True
        parts = [part for part in normalized.split("/") if part not in ("", ".")]
        return any(part == ".." for part in parts)

    @staticmethod
    def validate_zip_file(zip_path: str) -> tuple[bool, str, list[str]]:
        """
        Validate zip file and its contents
        Returns: (is_valid, error_message, valid_files)
        """
        try:
            # Check zip file size
            if os.path.getsize(zip_path) > ZipValidator.MAX_ZIP_SIZE:
                return False, "Zip file too large (max 100MB)", []

            valid_files = []
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Check number of files
                if len(zip_ref.filelist) > ZipValidator.MAX_FILES:
                    return False, f"Too many files in zip (max {ZipValidator.MAX_FILES})", []

                # Check for directory traversal attempts
                for file_info in zip_ref.filelist:
                    if ZipValidator._is_malicious_zip_path(file_info.filename):
                        return False, "Invalid file paths detected", []

                # Validate each file
                total_size = 0
                for file_info in zip_ref.filelist:
                    # Skip directories
                    if file_info.filename.endswith('/'):
                        continue

                    # Check file size
                    if file_info.file_size > ZipValidator.MAX_ZIP_SIZE:
                        return False, f"File {file_info.filename} too large", []

                    total_size += file_info.file_size
                    if total_size > ZipValidator.MAX_ZIP_SIZE:
                        return False, "Total uncompressed size too large", []

                    # Check file extension
                    ext = os.path.splitext(file_info.filename)[1].lower()
                    if ext in ZipValidator.VALID_EXTENSIONS:
                        valid_files.append(file_info.filename)

            if not valid_files:
                return False, "No valid markdown or text files found in zip", []

            return True, "", valid_files

        except zipfile.BadZipFile:
            return False, "Invalid or corrupted zip file", []
        except (OSError, RuntimeError, TypeError, ValueError, zipfile.LargeZipFile) as e:
            return False, f"Error processing zip file: {str(e)}", []

def format_text_with_line_breaks(text):
    """Insert HTML line breaks after sentence-ending punctuation."""
    # Split the text into sentences and add line breaks
    sentences = text.replace('. ', '.<br>').replace('? ', '?<br>').replace('! ', '!<br>')
    return sentences


def format_transcript(raw_text: str) -> str:
    """Convert timestamped transcript to readable format"""
    lines = []
    for line in raw_text.split('\n'):
        if '|' in line:
            timestamp, text = line.split('|', 1)
            lines.append(f"{text.strip()}")
        else:
            lines.append(line.strip())
    return '\n'.join(lines)

#
# End of File Handling Functions
#######################################################################################################################

def extract_media_id_from_result_string(result_msg: str | None) -> str | None:
    """
    Extracts the Media ID from a string expected to contain 'Media ID: <id>'.

    This function searches for the pattern "Media ID:" followed by optional
    whitespace and captures the subsequent sequence of non-whitespace characters
    as the ID.

    Args:
        result_msg: The input string potentially containing the Media ID message,
                    typically returned by processing functions like import_epub.

    Returns:
        The extracted Media ID as a string if the pattern is found.
        Returns None if the input string is None, empty, or the pattern
        "Media ID: <id>" is not found.

    Examples:
        >>> extract_media_id_from_result_string("Ebook imported successfully. Media ID: ebook_789")
        'ebook_789'
        >>> extract_media_id_from_result_string("Success. Media ID: db_mock_id")
        'db_mock_id'
        >>> extract_media_id_from_result_string("Error during processing.")
        None
        >>> extract_media_id_from_result_string(None)
        None
        >>> extract_media_id_from_result_string("Media ID: id-with-hyphens123") # Test hyphens/numbers
        'id-with-hyphens123'
        >>> extract_media_id_from_result_string("Media ID:id_no_space") # Test no space
        'id_no_space'
    """
    # Handle None or empty input string gracefully
    if not result_msg:
        return None

    # Regular expression pattern:
    # - Looks for the literal string "Media ID:" (case-sensitive).
    # - Allows for zero or more whitespace characters (\s*) after the colon.
    # - Captures (\(...\)) one or more non-whitespace characters (\S+).
    #   Using \S+ is generally safer than \w+ as IDs might contain hyphens or other symbols.
    #   If IDs are strictly alphanumeric + underscore, you could use (\w+) instead.
    # - We use re.search to find the pattern anywhere in the string.
    pattern = r"Media ID:\s*(\S+)"

    match = re.search(pattern, result_msg)

    # If a match is found, match.group(1) will contain the captured ID part
    if match:
        media_id = match.group(1).rstrip(".,;:)]}\"'")
        return media_id or None
    else:
        # The pattern "Media ID: <id>" was not found in the string
        return None

def is_valid_date(date_string): # Placeholder
    """Return whether a string is a YYYY-MM-DD date."""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False


#
# End of Utils.py
#######################################################################################################################
