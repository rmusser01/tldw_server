# Media_Wiki.py
# Description: This file contains the functions to import MediaWiki dumps into the media_db and Chroma databases.
#######################################################################################################################
#
# Imports
import bz2
import contextlib
import gzip
import json
import os
import re
import sys
import tempfile
import traceback
from collections.abc import Iterator
from datetime import datetime, timezone  # Added for default ingestion_date
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import quote

import mwparserfromhell
import mwxml
import yaml

#
# 3rd-Party Imports
from loguru import logger as _base_logger

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.media_db.api import get_media_repository
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database

#
# Local Imports
from tldw_Server_API.app.core.exceptions import InvalidStoragePathError
from tldw_Server_API.app.core.Utils.Utils import logging

try:  # Optional embeddings dependencies
    from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
        create_embeddings_batch,
    )
except (ImportError, AttributeError):  # pragma: no cover - embeddings optional
    ChromaDBManager = None  # type: ignore
    create_embeddings_batch = None  # type: ignore
#
#######################################################################################################################
#
# Functions:
# Load configuration
def load_mediawiki_import_config():
    # Build config path safely using Path
    base_dir = Path(__file__).parent.resolve()
    config_path = base_dir / '..' / '..' / '..' / '..' / 'Config_Files' / 'mediawiki_import_config.yaml'
    config_path = config_path.resolve()

    # Verify the path is within the project structure
    project_root = base_dir.parent.parent.parent.parent.resolve()
    if not str(config_path).startswith(str(project_root)):
        raise InvalidStoragePathError("Config file path is outside project directory")

    with open(config_path) as f:
        return yaml.safe_load(f)


logger = _base_logger.bind(component="mediawiki_import")
_STDOUT_HANDLER_ID: Optional[int] = None
_FILE_HANDLER_ID: Optional[int] = None
ALLOWED_MEDIAWIKI_DUMP_EXTENSIONS = frozenset({".xml", ".xml.bz2", ".xml.gz", ".bz2", ".gz"})
MAX_MEDIAWIKI_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10GB limit
MEDIAWIKI_RUNTIME_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RuntimeError,
    json.JSONDecodeError,
    InvalidStoragePathError,
)
MEDIAWIKI_EMBEDDING_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RuntimeError,
)


def get_safe_log_path(log_filename: str) -> Optional[Path]:
    """Generate safe log file path.

    Args:
        log_filename: Name of the log file

    Returns:
        Safe Path object for log file or None if unsafe
    """
    try:
        # Validate log filename
        if not re.match(r'^[a-zA-Z0-9_\-]+\.log$', log_filename):
            logger.warning(f"Invalid log filename: {log_filename}")
            return None

        # Check for path traversal attempts
        if any(pattern in log_filename for pattern in ['..', '/', '\\', '\x00']):
            logger.warning(f"Path traversal attempt in log filename: {log_filename}")
            return None

        # Build safe log directory path
        base_dir = Path(__file__).parent.resolve()
        project_root = base_dir.parent.parent.parent.parent.resolve()

        # Allow overriding the log directory via environment for test/CI
        env_log_dir = os.getenv('TLDB_LOG_DIR')
        if env_log_dir:
            candidate = Path(env_log_dir)
            if not candidate.is_absolute():
                candidate = (project_root / candidate).resolve()
            # Ensure the chosen directory remains within the project root for safety
            try:
                common = os.path.commonpath([str(candidate), str(project_root)])
                log_dir = candidate if common == str(project_root) else (project_root / 'Logs')
            except ValueError:
                log_dir = project_root / 'Logs'
        else:
            log_dir = project_root / 'Logs'

        # Create log directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)

        # Construct full log path
        log_path = (log_dir / log_filename).resolve()

        # Verify the path is within the log directory
        try:
            common_path = os.path.commonpath([str(log_path), str(log_dir)])
            if common_path != str(log_dir):
                logger.warning(f"Log file path outside log directory: {log_path}")
                return None
        except ValueError:
            logger.warning(f"Invalid log file path: {log_path}")
            return None

        return log_path
    except (OSError, ValueError, TypeError) as e:
        logger.exception(f"Error creating safe log path: {e}")
        return None


def setup_media_wiki_logger(name: str, level: Union[int, str] = "INFO", log_file: Optional[str] = None) -> None:
    """Set up the logger with the given name and level."""
    global logger, _STDOUT_HANDLER_ID, _FILE_HANDLER_ID
    logger = _base_logger.bind(component=name)

    if _STDOUT_HANDLER_ID is not None:
        _base_logger.remove(_STDOUT_HANDLER_ID)
    _STDOUT_HANDLER_ID = _base_logger.add(
        sys.stdout,
        format="{time} - {extra[component]} - {level} - {message}",
        level=level,
        filter=lambda record, n=name: record["extra"].get("component") == n,
    )

    # Optionally disable file-based logs (useful in sandboxed test runs)
    disable_file_logs = os.getenv('TLDB_DISABLE_FILE_LOGS', '0') == '1'
    if log_file and not disable_file_logs:
        safe_log_path = get_safe_log_path(log_file)
        if safe_log_path:
            if _FILE_HANDLER_ID is not None:
                _base_logger.remove(_FILE_HANDLER_ID)
            _FILE_HANDLER_ID = _base_logger.add(
                str(safe_log_path),
                format="{time} - {extra[component]} - {level} - {message}",
                level=level,
                filter=lambda record, n=name: record["extra"].get("component") == n,
            )
        else:
            logger.warning(f"Could not create log file: {log_file}. Logging to stdout only.")


setup_media_wiki_logger('mediawiki_import', log_file='mediawiki_import.log')

_mediawiki_import_config_cache: Optional[dict[str, Any]] = None


def get_mediawiki_import_config() -> dict[str, Any]:
    """Lazy-load and cache MediaWiki import config."""
    global _mediawiki_import_config_cache
    if _mediawiki_import_config_cache is None:
        try:
            _mediawiki_import_config_cache = load_mediawiki_import_config() or {}
        except (OSError, yaml.YAMLError, InvalidStoragePathError, ValueError, TypeError) as exc:
            logger.warning(f"Failed to load mediawiki_import_config.yaml, using defaults: {exc}")
            _mediawiki_import_config_cache = {}
    return _mediawiki_import_config_cache


#
#######################################################################################################################
#
# Functions:

def validate_file_path(file_path: str, allowed_dir: Optional[Path] = None) -> Path:
    """Validate and resolve file path to prevent path traversal attacks.

    Args:
        file_path: Path to validate
        allowed_dir: Optional directory to restrict file access to (default: current working directory)

    Returns:
        Resolved safe Path object

    Raises:
        InvalidStoragePathError: If path is invalid or attempts traversal
    """
    try:
        # Check for null bytes which can truncate paths
        if '\x00' in file_path:
            raise InvalidStoragePathError("Null byte in path")

        # Additional checks for suspicious patterns first
        if '../' in file_path or '..' + os.sep in file_path:
            raise InvalidStoragePathError("Path traversal attempt detected")

        # Convert to Path for symlink inspection before resolution
        orig_path = Path(file_path)

        # Default to current working directory if no allowed_dir specified
        if allowed_dir is None:
            # For MediaWiki dumps, we expect them to be in a reasonable location
            # Default to allowing files in current directory and subdirectories
            allowed_dir = Path.cwd()

        # Check for symlink attacks
        if orig_path.is_symlink():
            # Resolve the symlink and check if it's within allowed directory
            real_path = orig_path.resolve()
            allowed_resolved = Path(allowed_dir).resolve()
            try:
                common_path = os.path.commonpath([str(real_path), str(allowed_resolved)])
                if common_path != str(allowed_resolved):
                    raise InvalidStoragePathError("Symlink points outside allowed directory")
            except ValueError as exc:
                raise InvalidStoragePathError("Symlink points outside allowed directory") from exc

        # Resolve to absolute path for subsequent checks
        path = orig_path.resolve()

        # Check if path exists
        if not path.exists():
            raise InvalidStoragePathError("File does not exist")

        # Check if it's a file (not a directory or symlink to directory)
        if not path.is_file():
            raise InvalidStoragePathError("Path is not a regular file")

        allowed = Path(allowed_dir).resolve()
        # Use os.path.commonpath for secure path containment check
        try:
            common_path = os.path.commonpath([str(path), str(allowed)])
            if common_path != str(allowed):
                raise InvalidStoragePathError("Access denied: Path is outside allowed directory")
        except ValueError as exc:
            # Paths are on different drives (Windows) or otherwise incomparable
            raise InvalidStoragePathError("Access denied: Path is outside allowed directory") from exc

        lower_path = str(path).lower()
        if not any(lower_path.endswith(ext.lower()) for ext in ALLOWED_MEDIAWIKI_DUMP_EXTENSIONS):
            raise InvalidStoragePathError("File extension is not allowed")

        # Check file size to prevent processing huge files
        max_file_size = MAX_MEDIAWIKI_FILE_SIZE_BYTES
        if path.stat().st_size > max_file_size:
            raise InvalidStoragePathError("File size exceeds maximum allowed size")

        return path
    except InvalidStoragePathError as e:
        logger.exception(f"Path validation failed: {e}")
        raise
    except OSError as e:
        # Log the error internally but don't expose the path in the external message.
        logger.exception(f"Path validation failed: {e}")
        raise InvalidStoragePathError(
            f"Invalid file path: {str(e).replace(file_path, '[REDACTED]')}"
        ) from e


def sanitize_wiki_name(wiki_name: str) -> str:
    """Sanitize wiki name to prevent path traversal and injection attacks.

    Args:
        wiki_name: Wiki name to sanitize

    Returns:
        Sanitized wiki name

    Raises:
        ValueError: If wiki name contains invalid characters
    """
    # Check for null bytes first
    if '\x00' in wiki_name:
        raise ValueError("Invalid wiki name: Contains null byte")

    # Only allow alphanumeric, underscore, hyphen, and spaces
    if not re.match(r'^[a-zA-Z0-9_\- ]+$', wiki_name):
        raise ValueError("Invalid wiki name: Only alphanumeric characters, underscores, hyphens, and spaces are allowed")

    # Additional security checks
    if any(pattern in wiki_name for pattern in ['..', '/', '\\']):
        raise ValueError("Invalid wiki name: Contains forbidden characters")

    # Replace spaces with underscores for filesystem safety
    safe_name = wiki_name.replace(' ', '_')

    # Limit length to prevent issues
    if len(safe_name) > 100:
        raise ValueError("Wiki name too long (max 100 characters)")

    return safe_name


def get_safe_checkpoint_path(wiki_name: str, checkpoint_dir: Optional[Path] = None) -> Path:
    """Generate safe checkpoint file path.

    Args:
        wiki_name: Wiki name for checkpoint
        checkpoint_dir: Directory for checkpoint files (default: current directory)

    Returns:
        Safe Path object for checkpoint file
    """
    # Sanitize wiki name first
    safe_wiki_name = sanitize_wiki_name(wiki_name)

    # Use provided directory or default to a safe location
    if checkpoint_dir:
        base_dir = Path(checkpoint_dir).resolve()
    else:
        # Default to a checkpoints subdirectory
        base_dir = Path('./checkpoints').resolve()
        base_dir.mkdir(exist_ok=True)

    # Construct checkpoint filename
    checkpoint_filename = f"{safe_wiki_name}_import_checkpoint.json"
    checkpoint_path = base_dir / checkpoint_filename

    # Verify the path is within the expected directory using secure method
    try:
        # Use os.path.commonpath for secure path containment check
        checkpoint_resolved = checkpoint_path.resolve()
        base_resolved = base_dir.resolve()
        common_path = os.path.commonpath([str(checkpoint_resolved), str(base_resolved)])
        if common_path != str(base_resolved):
            raise InvalidStoragePathError("Invalid checkpoint path")
    except ValueError as exc:
        # Paths are on different drives (Windows) or otherwise incomparable
        raise InvalidStoragePathError("Invalid checkpoint path") from exc

    return checkpoint_path


def _open_dump_file_text(safe_path: Path):
    """Open a MediaWiki dump file as text, supporting .xml, .xml.bz2, .xml.gz.

    Returns a file-like object opened in text mode with utf-8 encoding.
    """
    lower = str(safe_path).lower()
    if lower.endswith('.xml.bz2') or lower.endswith('.bz2'):
        return bz2.open(safe_path, mode='rt', encoding='utf-8', errors='ignore')
    if lower.endswith('.xml.gz') or lower.endswith('.gz'):
        return gzip.open(safe_path, mode='rt', encoding='utf-8', errors='ignore')
    # Default to plain XML
    return open(safe_path, encoding='utf-8', errors='ignore')


def parse_mediawiki_dump(
    file_path: Union[str, Path],
    namespaces: Optional[list[int]] = None,
    skip_redirects: bool = False,
    allowed_dir: Optional[Path] = None,
    *,
    prevalidated_path: Optional[Path] = None,
) -> Iterator[dict[str, Any]]:
    # Validate file path
    safe_path = prevalidated_path if prevalidated_path is not None else validate_file_path(
        str(file_path),
        allowed_dir=allowed_dir,
    )
    # Use context manager for file operations to prevent resource leaks
    with _open_dump_file_text(safe_path) as f:
        dump = mwxml.Dump.from_file(f)
        for page in dump.pages:
            if skip_redirects and page.redirect:
                continue
            if namespaces and page.namespace not in namespaces:
                continue

            for revision in page:  # mwxml revisions are an iterator
                wikicode = mwparserfromhell.parse(revision.text or "")  # Ensure text is not None
                plain_text = wikicode.strip_code()
                # Normalize timestamp to a timezone-aware datetime when possible
                _ts = getattr(revision, "timestamp", None)
                if isinstance(_ts, datetime):
                    timestamp_obj = _ts.replace(tzinfo=timezone.utc) if _ts.tzinfo is None else _ts
                elif _ts is not None:
                    # Attempt to parse from string representation (e.g., 'YYYY-MM-DDTHH:MM:SSZ')
                    try:
                        ts_str = str(_ts)
                        ts_str = ts_str.replace('Z', '+00:00')
                        timestamp_obj = datetime.fromisoformat(ts_str)
                        if timestamp_obj.tzinfo is None:
                            timestamp_obj = timestamp_obj.replace(tzinfo=timezone.utc)
                    except (TypeError, ValueError):
                        timestamp_obj = datetime.now(timezone.utc)
                else:
                    # Fallback timestamp if revision has none
                    timestamp_obj = datetime.now(timezone.utc)

                yield {
                    "title": page.title,
                    "content": plain_text,
                    "namespace": page.namespace,
                    "page_id": page.id,
                    "revision_id": revision.id,
                    "timestamp": timestamp_obj  # Store as datetime object
                }
            logging.debug(f"Yielded page: {page.title}")


def optimized_chunking(text: str, chunk_options: dict[str, Any]) -> list[dict[str, Any]]:
    # Using simple newline splitting for sections as an example.
    # Your original implementation used re.split(r'\n==\s*(.*?)\s*==\n', text)
    # which is good for MediaWiki section syntax.
    # This function should produce a list of dictionaries, e.g.,
    # [{"text": "chunk text 1", "metadata": {"section_title": "Introduction", ...}}, ...]

    cfg = get_mediawiki_import_config()
    max_size = chunk_options.get('max_size', cfg.get('chunking', {}).get('default_size', 1000))
    # Fallback to simple splitting if no section-based logic is defined or needed
    # For this example, we'll just split by paragraphs if no section logic from original needed.
    # Your original `optimized_chunking` was section-aware. Let's keep that spirit.
    # If `text` is None or empty, handle gracefully
    if not text:
        return []

    sections = re.split(r'(\n==\s*[^=]+?\s*==\n)', text)  # Keep delimiters
    chunks = []
    current_chunk_text = ""
    current_section_title = "Introduction"  # Default for content before first heading

    # If the text doesn't start with a section, the first part is 'Introduction'
    if sections and not sections[0].startswith("\n=="):
        first_content = sections.pop(0).strip()
        if first_content:
            # If current_chunk_text + first_content is too large, start a new chunk
            if len(current_chunk_text) + len(first_content) > max_size and current_chunk_text:
                chunks.append({"text": current_chunk_text.strip(), "metadata": {"section": current_section_title}})
                current_chunk_text = first_content
            else:
                current_chunk_text += ("\n" if current_chunk_text else "") + first_content
    else:  # Text might start with a delimiter or be empty
        if sections and sections[0].strip() == "":  # Handle empty first element from split
            sections.pop(0)

    for i in range(0, len(sections), 2):
        header_part = sections[i].strip() if i < len(sections) else ""
        content_part = sections[i + 1].strip() if i + 1 < len(sections) else ""

        if header_part.startswith("==") and header_part.endswith("=="):
            new_section_title = header_part.strip("= \n")
            if current_chunk_text.strip():  # If there's content for the previous section, store it
                chunks.append({"text": current_chunk_text.strip(), "metadata": {"section": current_section_title}})
            current_chunk_text = ""  # Reset for new section
            current_section_title = new_section_title

        if content_part:  # Add content to current section's chunk
            # Further split content_part if it alone exceeds max_size (simplified here)
            if len(current_chunk_text) + len(content_part) > max_size and current_chunk_text.strip():
                chunks.append({"text": current_chunk_text.strip(), "metadata": {"section": current_section_title}})
                current_chunk_text = content_part
            else:
                current_chunk_text += ("\n" if current_chunk_text else "") + content_part

    # Add any remaining text
    if current_chunk_text.strip():
        chunks.append({"text": current_chunk_text.strip(), "metadata": {"section": current_section_title}})

    # If no chunks were created (e.g. empty input text), return empty list
    if not chunks and text.strip():  # If text was not empty but no chunks (edge case)
        chunks.append({"text": text.strip(), "metadata": {"section": "Full Text"}})

    logging.debug(f"optimized_chunking: Created {len(chunks)} chunks.")
    return chunks


def _normalize_embedding_provider(provider: str) -> str:
    mapping = {
        "local": "local_api",
        "local_api": "local_api",
        "huggingface": "huggingface",
        "hf": "huggingface",
        "openai": "openai",
        "onnx": "onnx",
    }
    return mapping.get(provider.strip().lower(), provider.strip().lower())


def _build_mediawiki_embedding_config(
    *,
    api_name_vector_db: Optional[str],
    api_key_vector_db: Optional[str],
) -> Optional[dict[str, Any]]:
    cfg = get_mediawiki_import_config()
    embeddings_cfg = cfg.get("embeddings", {}) or {}
    provider = str(embeddings_cfg.get("provider") or "openai").strip()
    model = str(embeddings_cfg.get("model") or "text-embedding-3-small").strip()
    api_key = api_key_vector_db or embeddings_cfg.get("api_key")
    api_url = embeddings_cfg.get("local_url") or embeddings_cfg.get("api_url")

    if api_name_vector_db:
        override = str(api_name_vector_db).strip()
        if override:
            override_lower = override.lower()
            if override_lower in {"openai", "huggingface", "hf", "local", "local_api", "onnx"}:
                provider = override
            elif ":" in override:
                provider_part, model_part = override.split(":", 1)
                if provider_part.strip():
                    provider = provider_part.strip()
                if model_part.strip():
                    model = model_part.strip()
            else:
                model = override

    provider_norm = _normalize_embedding_provider(provider)
    if not provider_norm:
        logger.warning("MediaWiki embeddings provider not configured; skipping vector storage.")
        return None

    model_id = f"{provider_norm}:{model}"

    if provider_norm == "openai":
        model_cfg: dict[str, Any] = {
            "provider": "openai",
            "model_name_or_path": model,
        }
        if api_key:
            model_cfg["api_key"] = api_key
    elif provider_norm == "local_api":
        if not api_url:
            logger.warning("Local embeddings provider configured without api_url; skipping vector storage.")
            return None
        model_cfg = {
            "provider": "local_api",
            "model_name_or_path": model,
            "api_url": api_url,
        }
        if api_key:
            model_cfg["api_key"] = api_key
    elif provider_norm == "huggingface":
        model_cfg = {
            "provider": "huggingface",
            "model_name_or_path": model,
        }
    elif provider_norm == "onnx":
        model_cfg = {
            "provider": "onnx",
            "model_name_or_path": model,
        }
    else:
        logger.warning(
            "Unsupported embeddings provider '{}' for MediaWiki vector storage.",
            provider_norm,
        )
        return None

    return {
        "default_model_id": model_id,
        "models": {model_id: model_cfg},
    }


def _mediawiki_collection_name(wiki_name: str) -> str:
    cfg = get_mediawiki_import_config()
    prefix = (cfg.get("chromadb", {}) or {}).get("collection_prefix", "mediawiki_")
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(prefix)).strip()
    safe_wiki = sanitize_wiki_name(wiki_name)
    if safe_prefix:
        return f"{safe_prefix}{safe_wiki}"
    return safe_wiki


def _store_mediawiki_chunks_in_vector_db(
    *,
    chunks: list[dict[str, Any]],
    media_id: int,
    title: str,
    wiki_name: str,
    page_id: Optional[int],
    revision_id: Optional[int],
    api_name_vector_db: Optional[str],
    api_key_vector_db: Optional[str],
) -> tuple[bool, str]:
    if ChromaDBManager is None or create_embeddings_batch is None:
        return False, "Embeddings dependencies unavailable."

    embedding_config = _build_mediawiki_embedding_config(
        api_name_vector_db=api_name_vector_db,
        api_key_vector_db=api_key_vector_db,
    )
    if not embedding_config:
        return False, "Embeddings configuration missing or invalid."

    user_db_base_dir = settings.get("USER_DB_BASE_DIR")
    if not user_db_base_dir:
        return False, "USER_DB_BASE_DIR is not configured."

    user_embedding_config: dict[str, Any] = {
        "USER_DB_BASE_DIR": user_db_base_dir,
        "embedding_config": embedding_config,
    }

    vector_user_id = settings.get("SINGLE_USER_FIXED_ID", 1)
    try:
        manager = ChromaDBManager(
            user_id=str(vector_user_id),
            user_embedding_config=user_embedding_config,
        )
    except MEDIAWIKI_EMBEDDING_EXCEPTIONS as exc:
        return False, f"Vector store init failed: {exc}"

    indexed_chunks = [
        (idx, ch)
        for idx, ch in enumerate(chunks)
        if isinstance(ch, dict) and str(ch.get("text") or "").strip()
    ]
    if not indexed_chunks:
        return False, "No chunk text available for embeddings."

    texts = [str(ch.get("text")) for _, ch in indexed_chunks]
    model_id = embedding_config.get("default_model_id")
    try:
        embeddings = create_embeddings_batch(
            texts=texts,
            user_app_config=user_embedding_config,
            model_id_override=model_id,
        )
    except MEDIAWIKI_EMBEDDING_EXCEPTIONS as exc:
        return False, f"Embedding generation failed: {exc}"

    if not embeddings:
        return False, "Embedding generation returned no vectors."
    if len(embeddings) != len(texts):
        return False, "Embedding generation returned a mismatched vector count."

    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    for idx, ch in indexed_chunks:
        section = None
        metadata = ch.get("metadata") if isinstance(ch.get("metadata"), dict) else {}
        if isinstance(metadata, dict):
            section = metadata.get("section")
        metadatas.append(
            {
                "media_id": str(media_id),
                "page_id": page_id,
                "revision_id": revision_id,
                "wiki_name": wiki_name,
                "title": title,
                "chunk_index": idx,
                "section": section,
            }
        )
        ids.append(f"mediawiki_{media_id}_chunk_{idx}")

    collection_name = _mediawiki_collection_name(wiki_name)
    try:
        manager.store_in_chroma(
            collection_name=collection_name,
            texts=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
            embedding_model_id_for_dim_check=model_id,
        )
    except MEDIAWIKI_EMBEDDING_EXCEPTIONS as exc:
        return False, f"Vector store write failed: {exc}"

    return True, f"Stored {len(texts)} chunks in vector store '{collection_name}'."


def process_single_item(
        content: str,
        title: str,
        wiki_name: str,
        chunk_options: dict[str, Any],
        item: dict[str, Any],  # Contains timestamp, page_id etc. from parse_mediawiki_dump
        store_to_db: bool = True,
        store_to_vector_db: bool = True,
        api_name_vector_db: Optional[str] = None,
        api_key_vector_db: Optional[str] = None,
        media_writer: Any | None = None,
) -> dict[str, Any]:
    try:
        logging.debug(
            f"process_single_item: Processing item: {title} (StoreDB: {store_to_db}, StoreVector: {store_to_vector_db})")

        # Ensure timestamp is a datetime object for strftime, or handle if it's already string
        timestamp_dt = item.get("timestamp")
        if isinstance(timestamp_dt, str):
            try:
                timestamp_dt = datetime.fromisoformat(timestamp_dt.replace('Z', '+00:00'))
            except ValueError:
                timestamp_dt = datetime.now(timezone.utc)  # Fallback
        elif not isinstance(timestamp_dt, datetime):
            timestamp_dt = datetime.now(timezone.utc)  # Fallback

        iso_timestamp_str = timestamp_dt.isoformat()

        processed_data = {
            "title": title,
            "content": content,
            "namespace": item.get("namespace"),
            "page_id": item.get("page_id"),
            "revision_id": item.get("revision_id"),
            "timestamp": iso_timestamp_str,
            "chunks": [],
            "media_id": None,
            "message": "",
            "status": "Pending"
        }

        chunks = optimized_chunking(content, chunk_options)
        processed_data["chunks"] = chunks

        media_id = None
        if store_to_db:
            # Use URL-safe encoding to prevent injection attacks
            encoded_title = quote(title, safe='')
            # Sanitize wiki_name as well (already sanitized in import_mediawiki_dump, but double-check)
            safe_wiki_name = sanitize_wiki_name(wiki_name)
            url = f"mediawiki:{safe_wiki_name}:{encoded_title}"
            logging.debug(f"Generated Media URL: {url}")

            # Ensure ingestion_date is a string in 'YYYY-MM-DD' format
            ingestion_date_str = timestamp_dt.strftime('%Y-%m-%d')

            if media_writer is None:
                with managed_media_database(client_id="mediawiki_import") as db_instance:
                    result = get_media_repository(db_instance).add_media_with_keywords(
                        url=url,
                        title=title,
                        media_type="mediawiki_page",  # Adjusted type
                        content=content,
                        keywords=["mediawiki", wiki_name, "page"],
                        prompt="",
                        analysis_content="",  # Analysis/summary would be separate
                        transcription_model="N/A",
                        author="MediaWiki",  # Or parse from page if possible
                        ingestion_date=ingestion_date_str
                    )
            else:
                result = media_writer.add_media_with_keywords(
                    url=url,
                    title=title,
                    media_type="mediawiki_page",  # Adjusted type
                    content=content,
                    keywords=["mediawiki", wiki_name, "page"],
                    prompt="",
                    analysis_content="",  # Analysis/summary would be separate
                    transcription_model="N/A",
                    author="MediaWiki",  # Or parse from page if possible
                    ingestion_date=ingestion_date_str
                )
            # Assuming add_media_with_keywords returns (media_id, message)
            # If it returns the full DB record or just ID, adapt here.
            # Let's assume it's (media_id, message) as per your original code.
            if isinstance(result, tuple) and len(result) == 3:
                media_id, _, message = result
            elif isinstance(result, tuple) and len(result) == 2:
                media_id, message = result
            else:  # Fallback if structure is different
                media_id = result if isinstance(result, int) else None
                message = "DB operation status unknown" if media_id else "DB operation failed"

            processed_data["media_id"] = media_id
            processed_data["message"] = message
            logging.info(f"Media item DB result for '{title}': ID={media_id}, Msg='{message}'")

        vector_store_error = None
        if store_to_vector_db and media_id:
            success, message = _store_mediawiki_chunks_in_vector_db(
                chunks=chunks,
                media_id=media_id,
                title=title,
                wiki_name=wiki_name,
                page_id=item.get("page_id"),
                revision_id=item.get("revision_id"),
                api_name_vector_db=api_name_vector_db,
                api_key_vector_db=api_key_vector_db,
            )
            if success:
                processed_data["message"] += f" {message}"
            else:
                vector_store_error = message
                logging.warning(
                    "Vector storage skipped for '%s': %s",
                    title,
                    message,
                )
                processed_data["message"] += f" Skipped vector storage ({message})."
        elif store_to_vector_db and not media_id:
            logging.warning(
                f"Cannot store to vector DB for '{title}': media_id is missing (store_to_db may be False or failed).")
            processed_data["message"] += " Skipped vector storage (media_id missing)."

        if media_id or not store_to_db:
            processed_data["status"] = "Warning" if vector_store_error else "Success"
        else:
            processed_data["status"] = "Error"
        if media_id is None and store_to_db:  # If we intended to store but failed
            processed_data["message"] = processed_data.get("message", "") + " Failed to store media item to primary DB."

        logging.info(f"Successfully processed item '{title}' (Status: {processed_data['status']})")
        return processed_data

    except MEDIAWIKI_RUNTIME_EXCEPTIONS as e:
        logging.error(f"Error processing item {title}: {str(e)}")
        logging.error(f"Exception details: {traceback.format_exc()}")
        # Ensure all keys from 'processed_data' are present in error return
        timestamp_val = item.get("timestamp")
        if isinstance(timestamp_val, datetime):
            iso_timestamp_str_err = timestamp_val.isoformat()
        elif isinstance(timestamp_val, str):
            iso_timestamp_str_err = timestamp_val  # assume already iso
        else:
            iso_timestamp_str_err = datetime.now(timezone.utc).isoformat()

        return {
            "title": title, "status": "Error", "error_message": str(e), "chunks": [],
            "content": content,  # content might be available even if processing fails later
            "namespace": item.get("namespace"), "page_id": item.get("page_id"),
            "revision_id": item.get("revision_id"), "timestamp": iso_timestamp_str_err,
            "media_id": None, "message": f"Failed to process: {str(e)}"
        }


def load_checkpoint(file_path: str) -> int:
    """Load checkpoint from file.

    Args:
        file_path: Path to checkpoint file (must be within ./checkpoints directory)

    Returns:
        The last processed ID from the checkpoint, or 0 if not found/invalid
    """
    # Enforce checkpoints directory for defense-in-depth
    checkpoints_dir = Path('./checkpoints').resolve()

    # Check for null bytes and traversal patterns first
    if '\x00' in file_path or '../' in file_path or '..' + os.sep in file_path:
        logging.warning("Invalid checkpoint path attempted: [REDACTED]")
        return 0

    # Resolve and validate containment
    try:
        resolved_path = Path(file_path).resolve()
        common_path = os.path.commonpath([str(resolved_path), str(checkpoints_dir)])
        if common_path != str(checkpoints_dir):
            logging.warning("Checkpoint path outside checkpoints directory")
            return 0
    except ValueError:
        # Different drives on Windows or other path resolution issues
        return 0

    if not resolved_path.exists():
        return 0

    try:
        with open(resolved_path) as f:
            data = json.load(f)
            return data.get('last_processed_id', 0)
    except (json.JSONDecodeError, OSError):
        logging.warning("Checkpoint file is corrupted or unreadable. Starting from beginning.")
        return 0


def save_checkpoint(file_path: str, last_processed_id: int):
    # Check for null bytes
    if '\x00' in str(file_path):
        raise InvalidStoragePathError("Null byte in checkpoint path")

    # Validate the path is safe before any operations
    if '../' in str(file_path) or '..' + os.sep in str(file_path):
        raise InvalidStoragePathError("Path traversal attempt in checkpoint file")

    # Convert to Path and resolve
    safe_path = Path(file_path).resolve()

    # Ensure we're writing to the checkpoints directory
    checkpoints_dir = Path('./checkpoints').resolve()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Verify the resolved path is within checkpoints directory
    try:
        common_path = os.path.commonpath([str(safe_path), str(checkpoints_dir)])
        if common_path != str(checkpoints_dir):
            raise InvalidStoragePathError("Checkpoint file must be within checkpoints directory")
    except ValueError as exc:
        raise InvalidStoragePathError("Invalid checkpoint file path") from exc

    # Use atomic write to prevent partial writes and race conditions
    safe_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file first
    fd, temp_path = tempfile.mkstemp(dir=safe_path.parent, prefix='.tmp_', suffix='.json')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump({'last_processed_id': last_processed_id}, f)
        # Atomic rename (on POSIX systems)
        Path(temp_path).replace(safe_path)
    except Exception:
        # Clean up temp file on error
        with contextlib.suppress(OSError):
            os.unlink(temp_path)
        raise


def import_mediawiki_dump(
        file_path: str,
        wiki_name: str,
        namespaces: list[int] = None,
        skip_redirects: bool = False,
        chunk_options_override: dict[str, Any] = None,
        progress_callback: Any = None,
        store_to_db: bool = True,
        store_to_vector_db: bool = True,
        api_name_vector_db: Optional[str] = None,
        api_key_vector_db: Optional[str] = None,
        allowed_dir: Optional[Path] = None,
) -> Iterator[dict[str, Any]]:
    try:
        # Sanitize wiki_name and validate file_path
        safe_wiki_name = sanitize_wiki_name(wiki_name)
        # Restrict validation to the provided base directory if available
        resolved_allowed_dir = Path(allowed_dir).resolve() if allowed_dir is not None else None
        safe_file_path = validate_file_path(str(file_path), allowed_dir=resolved_allowed_dir)

        logging.info(
            f"Importing MediaWiki dump: {safe_file_path} for wiki: {safe_wiki_name}. StoreDB: {store_to_db}, StoreVector: {store_to_vector_db}")
        cfg = get_mediawiki_import_config()
        final_chunk_options = chunk_options_override if chunk_options_override else cfg.get('chunking', {})

        # Get safe checkpoint path
        checkpoint_file = get_safe_checkpoint_path(safe_wiki_name)
        last_processed_id = 0
        if store_to_db:  # Checkpoints only make sense if we are saving progress to DB
            last_processed_id = load_checkpoint(str(checkpoint_file))

        total_pages = count_pages(
            safe_file_path,
            namespaces,
            skip_redirects,
            allowed_dir=resolved_allowed_dir,
            prevalidated_path=safe_file_path,
        )
        processed_pages_count = 0

        yield {"type": "progress_total", "total_pages": total_pages,
               "message": f"Found {total_pages} pages to process for '{wiki_name}'."}

        with contextlib.ExitStack() as stack:
            shared_media_writer = None
            if store_to_db:
                db_instance = stack.enter_context(
                    managed_media_database(client_id="mediawiki_import")
                )
                shared_media_writer = get_media_repository(db_instance)

            for item_dict in parse_mediawiki_dump(
                safe_file_path,
                namespaces,
                skip_redirects,
                allowed_dir=resolved_allowed_dir,
                prevalidated_path=safe_file_path,
            ):
                current_page_id = item_dict.get('page_id', 0)
                current_title = item_dict.get('title', 'Unknown Title')

                if store_to_db and current_page_id <= last_processed_id:
                    processed_pages_count += 1
                    if progress_callback:
                        progress_callback(processed_pages_count / total_pages if total_pages > 0 else 0,
                                          f"Skipped (checkpoint): {current_title}")
                    yield {"type": "progress_item", "status": "skipped_checkpoint", "title": current_title,
                           "page_id": current_page_id,
                           "progress_percent": processed_pages_count / total_pages if total_pages > 0 else 0}
                    continue

                processed_item_details = process_single_item(
                    content=item_dict['content'],
                    title=current_title,
                    wiki_name=wiki_name,
                    chunk_options=final_chunk_options,
                    item=item_dict,  # Pass the full dict from parse_mediawiki_dump
                    store_to_db=store_to_db,
                    store_to_vector_db=store_to_vector_db,
                    api_name_vector_db=api_name_vector_db,
                    api_key_vector_db=api_key_vector_db,
                    media_writer=shared_media_writer,
                )

                if store_to_db and processed_item_details.get("status") == "Success" and processed_item_details.get(
                        "media_id") is not None:
                    save_checkpoint(str(checkpoint_file), current_page_id)

                processed_pages_count += 1
                current_progress_percent = processed_pages_count / total_pages if total_pages > 0 else 0
                if progress_callback:
                    progress_callback(current_progress_percent, f"Processed page: {current_title}")

                # Yield detailed result for each page, including its processing status
                yield {"type": "item_result", "data": processed_item_details, "progress_percent": current_progress_percent}

        if store_to_db and checkpoint_file.exists():
            try:
                # Validate checkpoint file path before deletion to ensure it's still safe
                checkpoint_resolved = checkpoint_file.resolve()
                checkpoints_dir = Path('./checkpoints').resolve()

                # Use os.path.commonpath to verify the file is still within expected directory
                try:
                    common_path = os.path.commonpath([str(checkpoint_resolved), str(checkpoints_dir)])
                    if common_path == str(checkpoints_dir):
                        checkpoint_file.unlink()
                        logging.info(f"Successfully removed checkpoint file: {checkpoint_file}")
                    else:
                        logging.warning(f"Checkpoint file {checkpoint_file} is outside expected directory, not removing")
                except ValueError:
                    # Paths are on different drives or incomparable
                    logging.warning(f"Checkpoint file {checkpoint_file} path validation failed, not removing")
            except OSError as e:
                logging.warning(f"Could not remove checkpoint file {checkpoint_file}: {e}")

        yield {"type": "summary",
               "message": f"Successfully processed MediaWiki dump: {wiki_name}. Processed {processed_pages_count}/{total_pages} pages."}

    except FileNotFoundError:
        logger.exception(f"MediaWiki dump file not found: {file_path}")
        yield {"type": "error", "message": f"Error: File not found - {file_path}"}
    except PermissionError:
        logger.exception(f"Permission denied when trying to read: {file_path}")
        yield {"type": "error", "message": f"Error: Permission denied - {file_path}"}
    except Exception as e:
        logger.exception(f"Error during MediaWiki import: {str(e)}")
        yield {"type": "error", "message": f"Error during import: {str(e)}"}


def count_pages(
    file_path: Union[str, Path],
    namespaces: Optional[list[int]] = None,
    skip_redirects: bool = False,
    allowed_dir: Optional[Path] = None,
    *,
    prevalidated_path: Optional[Path] = None,
) -> int:
    count = 0
    try:
        # Validate file path
        safe_path = prevalidated_path if prevalidated_path is not None else validate_file_path(
            str(file_path),
            allowed_dir=allowed_dir,
        )
        # Use context manager for file operations to prevent resource leaks
        with _open_dump_file_text(safe_path) as f:
            dump = mwxml.Dump.from_file(f)
            for page in dump.pages:
                if skip_redirects and page.redirect:
                    continue
                if namespaces and page.namespace not in namespaces:
                    continue
                count += 1
    except Exception as e:
        logger.error(f"Error counting pages in MediaWiki dump {file_path}: {str(e)}", exc_info=True)
        return 0  # Return 0 if counting fails
    return count

#
# End of Media_Wiki.py
#######################################################################################################################
