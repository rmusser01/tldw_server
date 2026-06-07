# database_retrievers.py
"""
Database-specific retrievers for the RAG service.

This module provides specialized retrieval strategies for different data sources,
including media database, notes, prompts, and character cards.
"""

import asyncio
import contextlib
import copy
from difflib import SequenceMatcher
import json
import os
import re
import sqlite3
import time
import urllib.parse as _urlparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.fts_translator import FTSQueryTranslator
from tldw_Server_API.app.core.DB_Management.Kanban_DB import KanbanDB
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    create_media_database,
    search_media,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    DatabaseError as MediaDatabaseError,
)

from .types import DataSource, Document
from .retrieval_plan import RetrievalPlan
from .utils import get_float_env as _get_float_env
from .utils import normalize_scores as _normalize_scores
from .vector_stores import (
    VectorStoreAdapter,
    create_from_settings_for_user,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class DatabasePathError(ValueError):
    """Raised when a database path is invalid."""

    def __init__(self, message: str = "Invalid database path") -> None:
        super().__init__(message)


class PathNormalizationError(DatabasePathError):
    """Raised when a database path cannot be normalized."""

    def __init__(self) -> None:
        super().__init__("Database path normalization failed")


class InvalidFileUriError(DatabasePathError):
    """Raised when a file:// URI is malformed."""

    def __init__(self) -> None:
        super().__init__("Invalid file URI for database path")


class UnsupportedDatabaseSchemeError(DatabasePathError):
    """Raised when a database path uses an unsupported URI scheme."""

    def __init__(self) -> None:
        super().__init__("Unsupported URI scheme in database path")


class PathTraversalError(DatabasePathError):
    """Raised when path traversal patterns are detected."""

    def __init__(self) -> None:
        super().__init__("Path traversal patterns (..) are not allowed")


class PathResolutionError(DatabasePathError):
    """Raised when a database path cannot be resolved."""

    def __init__(self) -> None:
        super().__init__("Database path resolution failed")


class SuspiciousDatabasePathError(DatabasePathError):
    """Raised when suspicious path patterns are detected."""

    def __init__(self) -> None:
        super().__init__("suspicious pattern detected in database path")


class RestrictedDatabasePathError(DatabasePathError):
    """Raised when a restricted directory is referenced."""

    def __init__(self) -> None:
        super().__init__("Access to restricted directory is not allowed")


class MissingDatabasePathError(DatabasePathError):
    """Raised when no database path is provided."""

    def __init__(self) -> None:
        super().__init__("db_path is required")


class RawSqlFallbackDisabledError(RuntimeError):
    """Raised when raw SQL fallback is disabled in production mode."""

    def __init__(self, retriever_name: str) -> None:
        super().__init__(
            f"Raw SQL fallback is disabled in production for {retriever_name}. Provide a database adapter."
        )


_RESIDUAL_ENCODED_PATH_CONTROL_RE = re.compile(r"%(?:2e|2f|5c)", re.IGNORECASE)


def _normalize_sqlite_memory_path(path: str) -> Optional[str]:
    """Return a canonical SQLite in-memory spec, or None when path is file-backed."""
    raw = str(path).strip()
    if not raw:
        return None

    lowered = raw.lower()
    if raw == ":memory:":
        return ":memory:"

    if lowered.startswith("file::memory:"):
        return raw

    if lowered.startswith("file:") and "mode=memory" in lowered:
        return raw

    if raw.startswith("file://"):
        try:
            parsed = _urlparse.urlparse(raw)
        except (TypeError, ValueError):
            return None
        extracted_path = _urlparse.unquote(parsed.path or "")
        query = parsed.query or ""
        if extracted_path in {":memory:", "/:memory:"}:
            return f"file::memory:?{query}" if query else ":memory:"
        if "mode=memory" in query.lower():
            return f"file::memory:?{query}" if query else ":memory:"

    return None


def _extract_file_uri_path(raw: str) -> str:
    """Extract the filesystem path from a file URI without query parameters."""
    parsed = _urlparse.urlparse(raw)
    extracted_path = _urlparse.unquote(parsed.path or "")
    netloc = _urlparse.unquote(parsed.netloc or "")
    if netloc:
        if re.match(r"^[A-Za-z]:", netloc):
            extracted_path = f"{netloc}{extracted_path}"
        elif extracted_path:
            extracted_path = f"//{netloc}{extracted_path}"
        else:
            extracted_path = netloc
    if re.match(r"^/[A-Za-z]:[/\\]", extracted_path):
        extracted_path = extracted_path[1:]
    return extracted_path


def _sanitize_media_fts_query(query: Optional[str]) -> Optional[str]:
    if query is None:
        return None
    try:
        text = str(query).strip()
    except (TypeError, ValueError):
        return query
    if not text:
        return text
    if text.startswith('"') and text.endswith('"'):
        return text
    if "-" in text and " " not in text:
        return f"\"{text}\""
    return text


_MEDIA_FALLBACK_STOP_WORDS = {
    "a",
    "about",
    "according",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "did",
    "do",
    "does",
    "during",
    "each",
    "explain",
    "explains",
    "fight",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "mention",
    "mentions",
    "new",
    "of",
    "on",
    "or",
    "issue",
    "issues",
    "reveal",
    "reveals",
    "said",
    "says",
    "tell",
    "than",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "toward",
    "under",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _derive_bounded_media_term_query(query: Optional[str], *, max_terms: int = 3) -> Optional[str]:
    if query is None:
        return None
    try:
        raw_text = str(query).strip()
    except (TypeError, ValueError):
        return None
    if not raw_text:
        return None

    candidates = re.findall(r"[A-Za-z0-9']+", raw_text.lower())
    filtered_terms: list[str] = []
    seen: set[str] = set()
    for token in candidates:
        normalized = token.strip("'")
        if len(normalized) < 4 or normalized in _MEDIA_FALLBACK_STOP_WORDS or normalized in seen:
            continue
        seen.add(normalized)
        filtered_terms.append(normalized)

    if not filtered_terms:
        return None

    if len(filtered_terms) <= max_terms:
        return " OR ".join(filtered_terms)

    selected_terms: set[str] = {filtered_terms[0], filtered_terms[-1]}
    middle_terms = filtered_terms[1:-1]
    if middle_terms and len(selected_terms) < max_terms:
        ranked_middle_terms = sorted(
            middle_terms,
            key=lambda term: (-len(term), filtered_terms.index(term)),
        )
        for term in ranked_middle_terms:
            selected_terms.add(term)
            if len(selected_terms) >= max_terms:
                break

    ordered_terms = [term for term in filtered_terms if term in selected_terms]
    return " OR ".join(ordered_terms)


def _extract_media_query_terms(query: Optional[str], *, max_terms: int = 8) -> list[str]:
    if query is None:
        return []
    try:
        raw_text = str(query).strip().lower()
    except (TypeError, ValueError):
        return []
    if not raw_text:
        return []

    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9']+", raw_text):
        normalized = token.strip("'")
        if len(normalized) < 3 or normalized in _MEDIA_FALLBACK_STOP_WORDS or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(normalized)
        if len(terms) >= max_terms:
            break
    return terms


def _score_media_chunk_text(
    chunk_text: str,
    query_terms: list[str],
    *,
    title: Optional[str] = None,
) -> float:
    lowered = chunk_text.lower()
    if not query_terms:
        return 0.0

    combined_text = f"{title or ''}\n{chunk_text}".lower()
    doc_terms = {
        token.strip("'")
        for token in re.findall(r"[A-Za-z0-9']+", combined_text)
        if len(token.strip("'")) >= 3
    }

    matched_score = 0.0
    total_occurrences = 0.0
    for term in query_terms:
        if term in combined_text:
            matched_score += 1.0
            total_occurrences += min(combined_text.count(term), 3) * 0.05
            continue

        if len(term) < 5:
            continue

        best_similarity = 0.0
        for doc_term in doc_terms:
            if abs(len(doc_term) - len(term)) > 2:
                continue
            similarity = SequenceMatcher(None, term, doc_term).ratio()
            if similarity > best_similarity:
                best_similarity = similarity
        if best_similarity >= 0.76:
            matched_score += 0.9 * best_similarity
            total_occurrences += 0.03

    return (matched_score / max(len(query_terms), 1)) + min(total_occurrences, 0.25)


_SQL_INPUT_PREFIX_RE = re.compile(
    r"^(?:\s|--[^\n]*\n|/\*.*?\*/)*(select|with)\b",
    flags=re.IGNORECASE | re.DOTALL,
)


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            try:
                return int(stripped)
            except (TypeError, ValueError):
                return None
    return None


def _get_metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata and metadata.get(key) is not None:
            return metadata.get(key)
        citation = metadata.get("citation")
        if isinstance(citation, dict) and citation.get(key) is not None:
            return citation.get(key)
        dotted_key = f"citation.{key}"
        if dotted_key in metadata and metadata.get(dotted_key) is not None:
            return metadata.get(dotted_key)
    return None


def _apply_location_metadata(document: Document) -> None:
    metadata = document.metadata or {}

    if document.start_char is None:
        start_val = _get_metadata_value(metadata, "start_char", "start_index", "start_offset", "paragraph_start")
        start_int = _coerce_int(start_val)
        if start_int is not None:
            document.start_char = start_int

    if document.end_char is None:
        end_val = _get_metadata_value(metadata, "end_char", "end_index", "end_offset", "paragraph_end")
        end_int = _coerce_int(end_val)
        if end_int is not None:
            document.end_char = end_int

    if document.chunk_index is None:
        chunk_val = _get_metadata_value(metadata, "chunk_index")
        chunk_int = _coerce_int(chunk_val)
        if chunk_int is not None:
            document.chunk_index = chunk_int

    if document.total_chunks is None:
        total_val = _get_metadata_value(metadata, "total_chunks")
        total_int = _coerce_int(total_val)
        if total_int is not None:
            document.total_chunks = total_int

    if document.page_number is None:
        page_val = _get_metadata_value(metadata, "page_number", "page", "page_no")
        page_int = _coerce_int(page_val)
        if page_int is not None:
            document.page_number = page_int

    if document.paragraph_number is None:
        para_val = _get_metadata_value(metadata, "paragraph_number", "paragraph")
        para_int = _coerce_int(para_val)
        if para_int is not None:
            document.paragraph_number = para_int

    if document.section_title is None:
        section_val = _get_metadata_value(metadata, "section_title", "section")
        if section_val is None:
            section_path = metadata.get("section_path")
            if isinstance(section_path, str) and section_path.strip():
                section_val = section_path.split(" > ")[-1]
        if isinstance(section_val, str) and section_val.strip():
            document.section_title = section_val.strip()


@dataclass
class RetrievalConfig:
    """Configuration for database retrieval."""
    max_results: int = 20
    min_score: float = 0.0
    use_fts: bool = True
    use_vector: bool = True
    include_metadata: bool = True
    date_filter: Optional[tuple[datetime, datetime]] = None
    tags_filter: Optional[list[str]] = None
    source_filter: Optional[list[str]] = None
    # FTS search level: media-level (default) or chunk-level (UnvectorizedMediaChunks)
    fts_level: Literal['media', 'chunk'] = 'media'
    # Query-scoped text late chunking: bypass stored chunks and rechunk
    # matched media in memory without persisting replacements.
    enable_text_late_chunking: bool = False
    chunk_method: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    chunk_language: Optional[str] = None


_RETRIEVAL_PLAN_SEARCH_MODES = frozenset({"fts", "vector", "hybrid"})
_RETRIEVAL_PLAN_SOURCE_ALIASES = {
    "character": DataSource.CHARACTER_CARDS,
    "characters": DataSource.CHARACTER_CARDS,
    "character_cards": DataSource.CHARACTER_CARDS,
    "character_cards_db": DataSource.CHARACTER_CARDS,
    "chats": DataSource.CHAT_HISTORY,
    "chat": DataSource.CHAT_HISTORY,
    "chat_history": DataSource.CHAT_HISTORY,
    "chat_history_db": DataSource.CHAT_HISTORY,
    "conversation": DataSource.CHAT_HISTORY,
    "conversations": DataSource.CHAT_HISTORY,
    "notes_db": DataSource.NOTES,
    "media": DataSource.MEDIA_DB,
    "media_db_path": DataSource.MEDIA_DB,
    "kanban_db": DataSource.KANBAN,
    "task_board": DataSource.KANBAN,
    "task_boards": DataSource.KANBAN,
    "tasks": DataSource.KANBAN,
    "prompt": DataSource.PROMPTS,
    "prompts_db": DataSource.PROMPTS,
    "worldbook": DataSource.WORLD_BOOKS,
    "worldbooks": DataSource.WORLD_BOOKS,
    "world_book": DataSource.WORLD_BOOKS,
    "world_books_db": DataSource.WORLD_BOOKS,
    "dictionary": DataSource.DICTIONARIES,
    "chat_dictionary": DataSource.DICTIONARIES,
    "chat_dictionaries": DataSource.DICTIONARIES,
    "chat_dictionaries_db": DataSource.DICTIONARIES,
}


def _normalize_plan_sources(plan: RetrievalPlan) -> list[DataSource]:
    normalized: list[DataSource] = []
    for raw_source in plan.sources:
        try:
            if isinstance(raw_source, DataSource):
                source = raw_source
            else:
                source_text = str(raw_source).strip().lower()
                source = _RETRIEVAL_PLAN_SOURCE_ALIASES.get(source_text) or DataSource(source_text)
        except (TypeError, ValueError):
            continue
        if source not in normalized:
            normalized.append(source)
    return normalized


def _config_from_retrieval_plan(
    config: Optional[RetrievalConfig],
    plan: RetrievalPlan,
) -> RetrievalConfig:
    effective = replace(config or RetrievalConfig())

    try:
        plan_top_k = int(plan.top_k)
    except (TypeError, ValueError):
        plan_top_k = int(effective.max_results or 10)
    effective.max_results = max(1, plan_top_k)

    try:
        effective.min_score = float(plan.min_score)
    except (TypeError, ValueError):
        pass

    search_mode = str(plan.search_mode or "").strip().lower()
    if search_mode in _RETRIEVAL_PLAN_SEARCH_MODES:
        effective.use_fts = search_mode in {"fts", "hybrid"}
        effective.use_vector = search_mode in {"vector", "hybrid"}

    return effective


def _index_namespace_from_retrieval_plan(
    plan: RetrievalPlan,
    sources: list[DataSource],
) -> Optional[str]:
    if plan.index_namespace is not None:
        return plan.index_namespace
    if DataSource.MEDIA_DB in sources:
        return plan.collection_names.get(DataSource.MEDIA_DB.value)
    return None


class BaseRetriever(ABC):
    """Base class for database-specific retrievers."""

    def __init__(
        self,
        db_path: Optional[str],
        config: Optional[RetrievalConfig] = None,
        *,
        db_adapter: Optional[Any] = None
    ) -> None:
        """Initialise the retriever with optional backend adapters."""
        self.config = config or RetrievalConfig()
        self._db_adapter = db_adapter
        self.db_path = self._validate_path(db_path) if db_path else None
        # Determine production mode from env var used across the project
        try:
            prod_val = str(os.getenv("tldw_production", "false")).strip().lower()
            self._production_mode = prod_val in {"true", "1", "yes", "on", "y"}
        except (AttributeError, TypeError, ValueError):
            self._production_mode = False
        # In production, prefer adapters over any raw SQL fallbacks
        if self._production_mode and self._db_adapter is None:
            logger.warning(
                f"Production mode active: no DB adapter provided for {self.__class__.__name__}. "
                "Raw SQL fallback is disabled and calls will fail without an adapter."
            )
        if self._db_adapter is None and self.db_path is None:
            raise MissingDatabasePathError()

    def _validate_path(self, path: Optional[str]) -> Optional[str]:
        """Validate and normalise database paths while guarding against traversal.

        Accepts strings or pathlib.Path instances for convenience.
        """
        if path is None:
            return None
        # Normalise non-string inputs (e.g., pathlib.Path) early
        try:
            if not isinstance(path, str):
                path = str(path)
        except (TypeError, ValueError) as exc:
            logger.error(f"Path normalization error for '{path}': {exc}")
            raise PathNormalizationError() from exc

        memory_path = _normalize_sqlite_memory_path(path)
        if memory_path is not None:
            return memory_path

        # Handle URI schemes - only allow file:// and validate the path component
        if '://' in path:
            if path.startswith('file://'):
                # Extract and decode the path component
                extracted_path = _extract_file_uri_path(path)
                extracted_parts = [part.lower() for part in extracted_path.split("/") if part]
                if (
                    extracted_path.startswith("/")
                    and extracted_parts
                    and extracted_parts[0] in {"etc", "proc", "sys", "dev", "boot", "root"}
                ):
                    raise RestrictedDatabasePathError()
                # Recursively validate the extracted path (this will catch traversal, etc.)
                validated = self._validate_path(extracted_path)
                if validated is None:
                    raise InvalidFileUriError()
                # Return only the validated absolute path - drop query params and URI scheme
                # This prevents SQLite URI mode options that could be dangerous
                return validated
            else:
                # Reject non-file URI schemes (http://, ftp://, etc.)
                scheme = path.split('://')[0]
                logger.warning(f"Rejected unsupported URI scheme: {scheme}")
                raise UnsupportedDatabaseSchemeError()

        # Check for path traversal sequences BEFORE resolving
        # This catches attempts like "../../../etc/passwd" before they get normalized
        if _RESIDUAL_ENCODED_PATH_CONTROL_RE.search(path):
            logger.warning(f"Residual encoded path-control sequence detected in: {path}")
            raise SuspiciousDatabasePathError()
        normalized_input_path = path.replace("\\", "/")
        if normalized_input_path.startswith("/"):
            first_component = normalized_input_path.split("/", 2)[1].lower()
            if first_component in {"etc", "proc", "sys", "dev", "boot", "root"}:
                raise RestrictedDatabasePathError()
        if '..' in path:
            logger.warning(f"Path traversal attempt detected in: {path}")
            raise PathTraversalError()

        try:
            path_obj = Path(path)
            abs_path = path_obj.resolve()
            path_str = str(abs_path)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error(f"Path validation error for '{path}': {exc}")
            raise PathResolutionError() from exc

        # Check for suspicious system paths in the resolved path
        suspicious_patterns = [
            '/etc/',
            '/proc/',
            '/sys/',
            '\\System32\\',
            '\\Windows\\',
        ]
        for pattern in suspicious_patterns:
            if pattern in path_str:
                logger.warning(f"Suspicious path pattern detected: {pattern} in {path_str}")
                raise SuspiciousDatabasePathError()
        if abs_path.parts and abs_path.parts[0] == '/' and len(abs_path.parts) > 1:
            restricted_dirs = ['etc', 'proc', 'sys', 'dev', 'boot', 'root']
            if abs_path.parts[1] in restricted_dirs:
                raise RestrictedDatabasePathError()
        parent_dir = abs_path.parent
        if not parent_dir.exists():
            logger.warning(f"Parent directory does not exist: {parent_dir}")
        return str(abs_path)

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve documents from database."""
        raise NotImplementedError

    @abstractmethod
    async def get_metadata(self, doc_id: str) -> dict[str, Any]:
        """Get metadata for a document."""
        raise NotImplementedError

    def _execute_query(
        self,
        query: str,
        params: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        """Execute SQL query and return results as dictionaries."""
        if self._db_adapter is not None:
            try:
                execute_query = getattr(self._db_adapter, "execute_query", None)
                if not callable(execute_query):
                    logger.error("DB adapter missing execute_query()")
                    return []
                cursor = execute_query(query, params)
                if cursor is None:
                    return []
                fetched = cursor.fetchall() or []
                return [dict(row) if not isinstance(row, dict) else row for row in fetched]
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as exc:
                logger.error(f"Backend query error: {exc}")
                return []
        # Disallow raw SQL fallback in production to honor project DB abstraction policy
        if getattr(self, "_production_mode", False):
            raise RawSqlFallbackDisabledError(self.__class__.__name__)
        if not self.db_path:
            logger.error("No database path available for direct query execution.")
            return []
        try:
            # Avoid logging raw SQL and params in production to reduce leakage risk
            if not getattr(self, "_production_mode", False):
                logger.debug(f"Executing query: {query[:100]}...")
                logger.debug(f"With params: {params}")
                logger.debug(f"Database path: {self.db_path}")
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                results = cursor.fetchall()
                logger.debug(f"Query returned {len(results)} results")
                return [dict(row) for row in results]
        except (sqlite3.Error, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.error(f"Database query error: {exc}")
            if not getattr(self, "_production_mode", False):
                logger.error(f"Query was: {query}")
                logger.error(f"Params were: {params}")
                logger.error(f"Database path: {self.db_path}")
            return []

    async def _execute_query_async(
        self,
        query: str,
        params: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        """Execute a synchronous backend query without blocking the event loop."""
        return await asyncio.to_thread(self._execute_query, query, params)


class MediaDBRetriever(BaseRetriever):
    """Retriever for Media_DB (main content database)."""

    def __init__(
        self,
        db_path: Optional[str],
        config: Optional[RetrievalConfig] = None,
        user_id: str = "0",
        media_db: Optional[Any] = None
    ) -> None:
        """Initialize MediaDBRetriever with optional vector store."""
        super().__init__(db_path, config, db_adapter=media_db)
        # Prefer an explicit adapter, otherwise try to attach the canonical media DB adapter.
        attached = None
        own = False
        if media_db is None:
            attached = self._maybe_attach_media_db(self.db_path)
            own = attached is not None
        self.media_db = media_db or attached
        self._db_adapter = self.media_db
        self._own_media_db = own
        self.user_id = user_id
        self.vector_store: Optional[VectorStoreAdapter] = None
        self._initialize_vector_store()

    def _maybe_attach_media_db(self, db_path: Optional[str]):
        """Best-effort: attach a content DB adapter when a canonical media DB path is provided.

        This enables robust retrieval in tests/CI where only the sqlite path is provided.
        """
        if not db_path:
            return None
        try:
            # Defensive re-validation in case callers bypass BaseRetriever.
            validated_path = self._validate_path(db_path)
            if not validated_path:
                return None
            return create_media_database("rag_service", db_path=validated_path)
        except MediaDatabaseError as exc:
            message = str(exc).lower()
            if "no such column" in message or "schema" in message:
                logger.debug("Skipping Media DB adapter attach for incompatible schema: {}", exc)
                return None
            raise
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            sqlite3.Error,
        ):
            return None

    def close(self):
        try:
            if self._own_media_db and self.media_db is not None:
                close_fn = getattr(self.media_db, 'close_connection', None)
                if callable(close_fn):
                    close_fn()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    def _initialize_vector_store(self):
        """Initialize vector store adapter if configured."""
        try:
            # Try to get vector store from settings
            from tldw_Server_API.app.core.config import settings
            settings_dict: dict[str, Any] = dict(settings)
            self.vector_store = create_from_settings_for_user(
                settings_dict,
                self.user_id
            )
            if self.vector_store is not None:
                logger.info(f"Vector store adapter initialized for MediaDBRetriever with user_id={self.user_id}")
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            logger.warning(f"Could not initialize vector store: {e}")
            self.vector_store = None

    async def retrieve(
        self,
        query: str,
        media_type: Optional[str] = None,
        **kwargs
    ) -> list[Document]:
        """Retrieve documents from the media database."""
        if self.media_db is not None:
            # Branch on FTS level when FTS search is enabled
            try:
                if self.config.use_fts and getattr(self.config, 'fts_level', 'media') == 'chunk':
                    if "enable_text_late_chunking" in kwargs:
                        prefer_text_late_chunking = bool(kwargs.get("enable_text_late_chunking"))
                    else:
                        prefer_text_late_chunking = bool(
                            getattr(self.config, "enable_text_late_chunking", False)
                        )
                    if prefer_text_late_chunking:
                        media_documents = self._retrieve_via_backend(
                            query,
                            media_type,
                            apply_min_score=False,
                            **kwargs,
                        )
                        late_chunk_documents = self._late_chunk_media_documents(
                            query,
                            media_documents,
                            **kwargs,
                        )
                        if late_chunk_documents:
                            return late_chunk_documents
                        return media_documents
                    chunk_documents, chunk_raw_count = self._retrieve_chunk_fts_with_stats(query, media_type, **kwargs)
                    if chunk_raw_count > 0:
                        return chunk_documents
                    media_documents = self._retrieve_via_backend(
                        query,
                        media_type,
                        apply_min_score=False,
                        **kwargs,
                    )
                    late_chunk_documents = self._late_chunk_media_documents(
                        query,
                        media_documents,
                        **kwargs,
                    )
                    if late_chunk_documents:
                        return late_chunk_documents
                    return media_documents
            except (AttributeError, RuntimeError, TypeError, ValueError):
                # Fall back gracefully to media-level
                pass
            return self._retrieve_via_backend(query, media_type, **kwargs)

        documents = []

        # Build FTS query
        fts_query = self._build_fts_query(query)

        # Column weights for bm25(title, content)
        title_w = 2.0
        content_w = 1.0
        try:
            from tldw_Server_API.app.core.config import settings as _settings
            # Allow nested RAG.FTS config or flat vars
            title_w = float((_settings.get("RAG", {}) or {}).get("fts_title_weight", _settings.get("FTS_TITLE_WEIGHT", 2.0)))
            content_w = float((_settings.get("RAG", {}) or {}).get("fts_content_weight", _settings.get("FTS_CONTENT_WEIGHT", 1.0)))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

        # Build SQL with filters
        sql = """
            SELECT
                m.id,
                m.title,
                m.content,
                m.type,
                m.url,
                m.ingestion_date,
                m.transcription_model,
                bm25(media_fts, ?, ?) as rank
            FROM media_fts
            JOIN media m ON media_fts.rowid = m.id
            WHERE media_fts MATCH ?
        """

        params = [title_w, content_w, fts_query]

        # Add media type filter
        if media_type:
            sql += " AND m.type = ?"
            params.append(media_type)

        # Add date filter
        if self.config.date_filter:
            start_date, end_date = self.config.date_filter
            sql += " AND m.ingestion_date BETWEEN ? AND ?"
            params.extend([start_date.isoformat(), end_date.isoformat()])

        # Optional restriction to specific media IDs
        allowed_media_ids = kwargs.get("allowed_media_ids")
        if allowed_media_ids and isinstance(allowed_media_ids, (list, tuple)):
            if not all(isinstance(mid, (int, str)) for mid in allowed_media_ids):
                raise ValueError("allowed_media_ids must be a list of ints or strings.")

            int_ids = [mid for mid in allowed_media_ids if isinstance(mid, int)]
            uuid_ids = [mid for mid in allowed_media_ids if isinstance(mid, str) and mid]
            filter_parts: list[str] = []

            if int_ids:
                placeholders = ",".join(["?"] * len(int_ids))
                filter_parts.append(f"m.id IN ({placeholders})")
                params.extend(int_ids)
            if uuid_ids:
                placeholders = ",".join(["?"] * len(uuid_ids))
                filter_parts.append(f"m.uuid IN ({placeholders})")
                params.extend(uuid_ids)
            if filter_parts:
                sql += f" AND ({' OR '.join(filter_parts)})"

        # Add ordering and limit (bm25: lower is better on SQLite)
        sql += " ORDER BY rank ASC LIMIT ?"
        params.append(self.config.max_results)

        # Execute query and normalize scores to [0,1] (higher is better)
        rows = list(self._execute_query(sql, tuple(params)))
        raw_scores = [float(r["rank"]) if r["rank"] is not None else 0.0 for r in rows]
        # bm25 (SQLite) is lower-better; invert before normalization
        inv_scores = [-s for s in raw_scores]
        norm = _normalize_scores(inv_scores, method="minmax") if rows else []

        # Convert to documents and apply min_score threshold on normalized scores
        min_score = float(self.config.min_score or 0.0)
        for row, score in zip(rows, norm):
            if score < min_score:
                continue
            doc = Document(
                id=str(row["id"]),
                content=row["content"] or "",
                source=DataSource.MEDIA_DB,
                metadata={
                    "title": row.get("title"),
                    "media_type": row.get("type"),
                    "url": row.get("url"),
                    "created_at": row.get("ingestion_date"),
                    "transcription_model": row.get("transcription_model"),
                    "source": "media_db"
                },
                score=float(score)
            )
            documents.append(doc)
        logger.debug(f"Retrieved {len(documents)} documents from Media_DB (normalized scores)")

        return documents

    def _retrieve_chunk_fts(self, query: str, media_type: Optional[str], **kwargs) -> list[Document]:
        documents, _raw_count = self._retrieve_chunk_fts_with_stats(query, media_type, **kwargs)
        return documents

    def _late_chunk_media_documents(
        self,
        query: str,
        media_documents: list[Document],
        **kwargs,
    ) -> list[Document]:
        if not media_documents:
            return []

        try:
            from tldw_Server_API.app.core.Chunking.chunker import Chunker  # type: ignore
        except ImportError:
            logger.debug("Late chunk media retrieval skipped because Chunker is unavailable")
            return []

        chunk_method = str(
            kwargs.get("chunk_method")
            or getattr(self.config, "chunk_method", None)
            or "sentences"
        )
        chunk_size = (
            _coerce_int(kwargs.get("chunk_size") or kwargs.get("max_chunk_size"))
            or _coerce_int(getattr(self.config, "chunk_size", None))
            or 500
        )
        chunk_overlap = (
            _coerce_int(kwargs.get("chunk_overlap"))
            or _coerce_int(getattr(self.config, "chunk_overlap", None))
            or 50
        )
        chunk_language = kwargs.get("chunk_language") or getattr(self.config, "chunk_language", None)
        query_terms = _extract_media_query_terms(query)

        try:
            chunker = Chunker()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug(f"Late chunk media retrieval skipped because Chunker could not initialize: {exc}")
            return []

        late_chunk_docs: list[Document] = []
        for parent_rank, media_doc in enumerate(media_documents[: max(1, self.config.max_results)]):
            parent_text = str(media_doc.content or "").strip()
            if not parent_text:
                continue

            try:
                flat_chunks = chunker.chunk_text_hierarchical_flat(
                    parent_text,
                    method=chunk_method,
                    max_size=chunk_size,
                    overlap=chunk_overlap,
                    language=chunk_language,
                )
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug(
                    "Late chunking failed for media document {}: {}",
                    media_doc.id,
                    exc,
                )
                continue

            if not isinstance(flat_chunks, list) or not flat_chunks:
                continue

            total_chunks = len(flat_chunks)
            parent_score = float(getattr(media_doc, "score", 0.0) or 0.0)
            for idx, chunk_item in enumerate(flat_chunks):
                if not isinstance(chunk_item, dict):
                    continue
                chunk_text = str(chunk_item.get("text") or "").strip()
                if not chunk_text:
                    continue

                chunk_metadata = chunk_item.get("metadata") or {}
                if not isinstance(chunk_metadata, dict):
                    chunk_metadata = {}

                md: dict[str, Any] = dict(media_doc.metadata or {})
                md.update(
                    {
                        "media_id": str(media_doc.id),
                        "chunk_index": idx,
                        "total_chunks": total_chunks,
                        "retrieval_mode": "late_chunk",
                        "source": "media_db",
                    }
                )

                for key in ("start_offset", "start_index"):
                    if key in chunk_metadata and chunk_metadata.get(key) is not None:
                        md["start_char"] = chunk_metadata.get(key)
                        break
                for key in ("end_offset", "end_index"):
                    if key in chunk_metadata and chunk_metadata.get(key) is not None:
                        md["end_char"] = chunk_metadata.get(key)
                        break
                for key in ("chunk_type", "paragraph_kind", "section_path", "ancestry_titles"):
                    if key in chunk_metadata and chunk_metadata.get(key) is not None and key not in md:
                        md[key] = chunk_metadata.get(key)

                chunk_score = _score_media_chunk_text(
                    chunk_text,
                    query_terms,
                    title=str(media_doc.metadata.get("title") or ""),
                )
                combined_score = (parent_score * 0.1) + (chunk_score * 0.9)
                # Earlier parent matches and earlier chunks should win ties.
                combined_score -= (parent_rank * 0.0001) + (idx * 0.00001)

                doc = Document(
                    id=f"late_chunk:{media_doc.id}:{idx}",
                    content=chunk_text,
                    source=DataSource.MEDIA_DB,
                    metadata=md,
                    score=combined_score,
                    source_document_id=str(media_doc.id),
                    source_document_metadata=dict(media_doc.metadata or {}),
                    parent_id=str(media_doc.id),
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    start_char=md.get("start_char"),
                    end_char=md.get("end_char"),
                )
                _apply_location_metadata(doc)
                late_chunk_docs.append(doc)

        late_chunk_docs.sort(key=lambda item: float(getattr(item, "score", 0.0)), reverse=True)
        return late_chunk_docs[: self.config.max_results]

    def _retrieve_chunk_fts_with_stats(
        self,
        query: str,
        media_type: Optional[str],
        **kwargs,
    ) -> tuple[list[Document], int]:
        """Retrieve chunk-level matches using FTS5 over UnvectorizedMediaChunks.

        For SQLite: uses virtual table `unvectorized_chunks_fts`.
        For Postgres: uses tsvector column on `unvectorized_media_chunks` (created via backend).
        """
        if self.media_db is None:
            return [], 0

        backend_type = getattr(self.media_db, 'backend_type', None)
        if backend_type == BackendType.SQLITE:
            # Ensure FTS virtual table exists; rebuild if empty to prime content
            try:
                ensure = getattr(self.media_db, 'ensure_chunk_fts', None)
                if callable(ensure):
                    self.media_db.ensure_chunk_fts()
                # Optionally prime FTS if empty (cheap count)
                check = getattr(self.media_db, 'maybe_rebuild_chunk_fts_if_empty', None)
                if callable(check):
                    self.media_db.maybe_rebuild_chunk_fts_if_empty()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug(f"Chunk FTS ensure/rebuild skipped: {exc}")

        params: list[Any] = []
        if backend_type == BackendType.SQLITE:
            # Build SQLite FTS query and SQL
            fts_query = self._build_fts_query(query)
            sql = """
                SELECT
                    u.uuid AS chunk_uuid,
                    u.id   AS chunk_rowid,
                    u.media_id,
                    u.chunk_text,
                    u.start_char,
                    u.end_char,
                    u.chunk_type,
                    u.chunk_index,
                    u.metadata AS chunk_metadata,
                    m.title,
                    m.type AS media_type,
                    m.url,
                    bm25(unvectorized_chunks_fts) AS rank
                FROM unvectorized_chunks_fts
                JOIN UnvectorizedMediaChunks u ON unvectorized_chunks_fts.rowid = u.id
                JOIN Media m ON u.media_id = m.id
                WHERE unvectorized_chunks_fts MATCH ?
                  AND m.deleted = 0 AND m.is_trash = 0 AND u.deleted = 0
            """
            params.append(fts_query)
        else:
            # Postgres tsquery path over generated tsvector column
            tsquery = FTSQueryTranslator.normalize_query(query, 'postgresql')
            sql = (
                "SELECT "
                " u.uuid AS chunk_uuid, u.id AS chunk_rowid, u.media_id, u.chunk_text,"
                " u.start_char, u.end_char, u.chunk_type, u.chunk_index, u.metadata AS chunk_metadata,"
                " m.title, m.type AS media_type, m.url,"
                " ts_rank(u.unvectorized_chunks_fts_tsv, to_tsquery('english', ?)) AS rank"
                " FROM unvectorizedmediachunks u"
                " JOIN media m ON u.media_id = m.id"
                " WHERE m.deleted = 0 AND m.is_trash = 0 AND u.deleted = 0"
                "   AND u.unvectorized_chunks_fts_tsv @@ to_tsquery('english', ?)"
            )
            params.extend([tsquery, tsquery])
        if media_type:
            sql += " AND m.type = ?"
            params.append(media_type)

        # Optional restriction to specific media IDs
        allowed_media_ids = kwargs.get("allowed_media_ids")
        if allowed_media_ids and isinstance(allowed_media_ids, (list, tuple)):
            if not all(isinstance(mid, (int, str)) for mid in allowed_media_ids):
                raise ValueError("allowed_media_ids must be a list of ints or strings.")

            int_ids = [mid for mid in allowed_media_ids if isinstance(mid, int)]
            uuid_ids = [mid for mid in allowed_media_ids if isinstance(mid, str) and mid]
            filter_parts: list[str] = []

            if int_ids:
                placeholders = ",".join(["?"] * len(int_ids))
                filter_parts.append(f"m.id IN ({placeholders})")
                params.extend(int_ids)
            if uuid_ids:
                placeholders = ",".join(["?"] * len(uuid_ids))
                filter_parts.append(f"m.uuid IN ({placeholders})")
                params.extend(uuid_ids)
            if filter_parts:
                sql += f" AND ({' OR '.join(filter_parts)})"

        # Optional date filter against Media.ingestion_date
        if self.config.date_filter:
            start_date, end_date = self.config.date_filter
            sql += " AND m.ingestion_date BETWEEN ? AND ?"
            params.extend([start_date.isoformat(), end_date.isoformat()])

        # Order by relevance: SQLite bm25 prefers ASC; Postgres ts_rank prefers DESC
        if backend_type == BackendType.SQLITE:
            sql += " ORDER BY rank ASC LIMIT ?"
        else:
            sql += " ORDER BY rank DESC LIMIT ?"
        params.append(self.config.max_results)

        try:
            execute_query = getattr(self.media_db, "execute_query", None)
            if not callable(execute_query):
                logger.error("Media DB adapter missing execute_query() for chunk FTS")
                return [], 0
            cursor = execute_query(sql, tuple(params))
            rows = cursor.fetchall() if cursor is not None else []
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.error(f"Chunk FTS query failed: {exc}")
            return [], 0

        docs: list[Document] = []

        # Normalize scores to [0,1] (higher is better)
        norm_scores: list[float] = []
        if rows:
            row_maps = [r if isinstance(r, dict) else dict(r) for r in rows]
            raw_vals: list[float] = []
            for rm in row_maps:
                rv = rm.get('rank')
                try:
                    raw_vals.append(float(rv) if rv is not None else 0.0)
                except (TypeError, ValueError):
                    raw_vals.append(0.0)
            if backend_type == BackendType.SQLITE:
                inv = [-v for v in raw_vals]
                norm_scores = _normalize_scores(inv, method="minmax")
            else:
                norm_scores = _normalize_scores(raw_vals, method="minmax")
        else:
            row_maps = []

        min_score = float(self.config.min_score or 0.0)
        for row, score_val in zip(row_maps, norm_scores):
            if float(score_val) < min_score:
                continue

            md: dict[str, Any] = {}
            if self.config.include_metadata:
                md = {
                    'title': row.get('title'),
                    'media_type': row.get('media_type'),
                    'url': row.get('url'),
                    'media_id': str(row.get('media_id')) if row.get('media_id') is not None else None,
                    'chunk_type': row.get('chunk_type'),
                    'chunk_index': int(row.get('chunk_index') or 0),
                    'start_char': row.get('start_char'),
                    'end_char': row.get('end_char'),
                    'source': 'media_db',
                }

                # Optionally enrich with nearest section info from DocumentStructureIndex
                try:
                    from tldw_Server_API.app.core.config import rag_enable_structure_index  # lazy import
                    _enable_si = rag_enable_structure_index()
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                    _enable_si = True
                if _enable_si:
                    try:
                        mid_raw = md.get('media_id')
                        st = md.get('start_char')
                        if self.media_db and mid_raw is not None and st is not None:
                            mid = int(str(mid_raw))
                            st_i = int(st)
                            lookup_section = getattr(self.media_db, "lookup_section_for_offset", None)
                            sec = lookup_section(mid, st_i) if callable(lookup_section) else None
                            if isinstance(sec, dict):
                                md['section_title'] = sec.get('title')
                                md['section_start'] = sec.get('start_char')
                                md['section_end'] = sec.get('end_char')
                                # Paragraph bounds default to chunk bounds
                                md.setdefault('paragraph_start', md.get('start_char'))
                                md.setdefault('paragraph_end', md.get('end_char'))
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        pass

                raw_meta = row.get("chunk_metadata")
                if raw_meta is not None:
                    extra_meta: Optional[dict[str, Any]] = None
                    try:
                        if isinstance(raw_meta, dict):
                            extra_meta = raw_meta
                        elif isinstance(raw_meta, (bytes, bytearray)):
                            extra_meta = json.loads(raw_meta.decode("utf-8", "ignore"))
                        elif isinstance(raw_meta, str):
                            extra_meta = json.loads(raw_meta)
                    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
                        extra_meta = None
                    if isinstance(extra_meta, dict):
                        for key, value in extra_meta.items():
                            if key not in md and value is not None:
                                md[key] = value

            chunk_uuid = str(row.get('chunk_uuid'))
            content_text = (row.get('chunk_text') or "")

            doc = Document(
                id=chunk_uuid,
                content=content_text,
                source=DataSource.MEDIA_DB,
                metadata=md,
                score=float(score_val),
                start_char=md.get('start_char'),
                end_char=md.get('end_char'),
                chunk_index=md.get('chunk_index'),
            )
            _apply_location_metadata(doc)
            docs.append(doc)

        return docs, len(rows)

    def _retrieve_via_backend(
        self,
        query: str,
        media_type: Optional[str],
        *,
        apply_min_score: bool = True,
        **kwargs,
    ) -> list[Document]:
        if self.media_db is None:
            return []
        date_range = None
        if self.config.date_filter:
            start, end = self.config.date_filter
            date_range = {'start_date': start, 'end_date': end}
        media_types = [media_type] if media_type else None
        sort_by = 'relevance' if self.config.use_fts else 'last_modified_desc'
        backend_type = getattr(self.media_db, 'backend_type', None)
        search_query = query
        if backend_type == BackendType.SQLITE:
            search_query = _sanitize_media_fts_query(query) or query
        results, raw_row_count = self._search_media_db(
            search_query=search_query,
            media_types=media_types,
            date_range=date_range,
            sort_by=sort_by,
            **kwargs,
        )
        if apply_min_score:
            documents = self._build_media_documents(results, backend_type=backend_type)
        else:
            documents = self._build_media_documents(
                results,
                backend_type=backend_type,
                apply_min_score=False,
            )
        if documents or not self.config.use_fts:
            return documents
        if raw_row_count > 0:
            return documents

        fallback_query = _derive_bounded_media_term_query(query)
        if not fallback_query:
            return documents
        if _sanitize_media_fts_query(fallback_query) == search_query:
            return documents
        return self._retrieve_media_term_fallback(
            fallback_query,
            media_type,
            apply_min_score=apply_min_score,
            **kwargs,
        )

    def _search_media_db(
        self,
        *,
        search_query: str,
        media_types: Optional[list[str]],
        date_range: Optional[dict[str, datetime]],
        sort_by: str,
        **kwargs,
    ) -> tuple[list[dict[str, Any]], int]:
        if self.media_db is None:
            return [], 0

        try:
            allowed_media_ids = kwargs.get("allowed_media_ids")
            results, _total = search_media(
                self.media_db,
                search_query=search_query,
                search_fields=['title', 'content'],
                media_types=media_types,
                date_range=date_range,
                media_ids_filter=list(allowed_media_ids) if allowed_media_ids else None,
                sort_by=sort_by,
                results_per_page=self.config.max_results,
                page=1,
                include_trash=False,
                include_deleted=False,
            )
        except (AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.error(f"MediaDatabase search failed: {exc}")
            return [], 0

        return results, len(results)

    def _build_media_documents(
        self,
        results: list[dict[str, Any]],
        *,
        backend_type: Any,
        apply_min_score: bool = True,
    ) -> list[Document]:
        documents: list[Document] = []

        # Normalize scores across results to [0,1] (higher is better)
        raw_vals: list[float] = []
        for row in results:
            rv = row.get('relevance_score')
            if rv is None:
                rv = row.get('rank')
            try:
                raw_vals.append(float(rv) if rv is not None else 0.0)
            except (TypeError, ValueError):
                raw_vals.append(0.0)
        if backend_type == BackendType.SQLITE:
            inv = [-v for v in raw_vals]
            norm_vals = _normalize_scores(inv, method="minmax") if raw_vals else []
        else:
            norm_vals = _normalize_scores(raw_vals, method="minmax") if raw_vals else []

        min_score = float(self.config.min_score or 0.0) if apply_min_score else 0.0
        for row, score_val in zip(results, norm_vals):
            if float(score_val) < min_score:
                continue
            metadata = {}
            if self.config.include_metadata:
                metadata = {
                    'title': row.get('title'),
                    'media_type': row.get('type'),
                    'url': row.get('url'),
                    'created_at': row.get('ingestion_date'),
                    'transcription_model': row.get('transcription_model'),
                    'last_modified': row.get('last_modified'),
                    'source': 'media_db',
                }
            # Use numeric media ID for Document.id to match callers/tests that
            # expect Media DB identifiers, and keep uuid in metadata if needed.
            doc_id = row.get('id')
            title_text = (row.get('title') or '').strip()
            body_text = (row.get('content') or '').strip()
            if title_text and (not body_text or title_text.lower() not in body_text.lower()):
                content_text = f"{title_text}\n{body_text}" if body_text else title_text
            else:
                content_text = body_text or title_text
            documents.append(
                Document(
                    id=str(doc_id),
                    content=content_text,
                    source=DataSource.MEDIA_DB,
                    metadata=metadata,
                    score=float(score_val),
                )
            )
        documents.sort(key=lambda doc: getattr(doc, 'score', 0.0), reverse=True)
        for doc in documents:
            _apply_location_metadata(doc)
        return documents

    def _retrieve_media_term_fallback(
        self,
        fallback_query: str,
        media_type: Optional[str],
        *,
        apply_min_score: bool = True,
        **kwargs,
    ) -> list[Document]:
        backend_type = getattr(self.media_db, 'backend_type', None)
        search_query = _sanitize_media_fts_query(fallback_query) or fallback_query
        results, _raw_row_count = self._search_media_db(
            search_query=search_query,
            media_types=[media_type] if media_type else None,
            date_range=self.config.date_filter and {
                'start_date': self.config.date_filter[0],
                'end_date': self.config.date_filter[1],
            },
            sort_by='relevance' if self.config.use_fts else 'last_modified_desc',
            **kwargs,
        )
        if apply_min_score:
            return self._build_media_documents(results, backend_type=backend_type)
        return self._build_media_documents(
            results,
            backend_type=backend_type,
            apply_min_score=False,
        )

    async def retrieve_with_keywords(
        self,
        query: str,
        keywords: list[str]
    ) -> list[Document]:
        """Retrieve with additional keyword filtering."""
        # Get base results
        documents = await self.retrieve(query)

        # Filter by keywords
        if keywords:
            filtered_docs = []
            for doc in documents:
                content_lower = doc.content.lower()
                if any(keyword.lower() in content_lower for keyword in keywords):
                    filtered_docs.append(doc)
            documents = filtered_docs

        return documents

    async def _retrieve_fts(
        self,
        query: str,
        media_type: Optional[str] = None,
        **kwargs
    ) -> list[Document]:
        """Internal method for FTS retrieval (same as retrieve)."""
        return await self.retrieve(query, media_type, **kwargs)

    @staticmethod
    def _normalize_allowed_media_ids_for_vector_filter(
        allowed_media_ids: Any,
    ) -> list[str]:
        if not isinstance(allowed_media_ids, (list, tuple)):
            return []

        normalized: list[str] = []
        for candidate in allowed_media_ids:
            try:
                media_id_str = str(int(candidate))
            except (TypeError, ValueError):
                continue
            if media_id_str not in normalized:
                normalized.append(media_id_str)
        return normalized

    @staticmethod
    def _build_allowed_media_vector_filter(
        allowed_media_ids: Any,
    ) -> Optional[dict[str, Any]]:
        normalized_ids = MediaDBRetriever._normalize_allowed_media_ids_for_vector_filter(
            allowed_media_ids
        )
        if not normalized_ids:
            return None
        if len(normalized_ids) == 1:
            return {"media_id": normalized_ids[0]}
        return {"media_id": {"$in": normalized_ids}}

    @staticmethod
    def _merge_vector_filters(
        base_filter: Optional[dict[str, Any]],
        scoped_filter: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        if not base_filter:
            return scoped_filter or {}
        if not scoped_filter:
            return base_filter
        return {"$and": [base_filter, scoped_filter]}

    @staticmethod
    def _extract_collection_metadatas(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        raw_metadatas = payload.get("metadatas")
        if not isinstance(raw_metadatas, list):
            return []
        if raw_metadatas and isinstance(raw_metadatas[0], list):
            flattened: list[dict[str, Any]] = []
            for block in raw_metadatas:
                if not isinstance(block, list):
                    continue
                flattened.extend(item for item in block if isinstance(item, dict))
            return flattened
        return [item for item in raw_metadatas if isinstance(item, dict)]

    def _resolve_scoped_query_embedding_override(
        self,
        *,
        collection_name: Optional[str],
        allowed_media_ids: Any,
    ) -> Optional[str]:
        if not collection_name:
            return None

        scoped_filter = self._build_allowed_media_vector_filter(allowed_media_ids)
        scoped_media_ids = set(
            self._normalize_allowed_media_ids_for_vector_filter(allowed_media_ids)
        )
        if not scoped_filter or not scoped_media_ids:
            return None

        vector_store = self.vector_store
        manager = getattr(vector_store, "manager", None) if vector_store is not None else None
        if manager is None:
            return None

        get_collection = getattr(manager, "get_collection", None)
        get_or_create_collection = getattr(manager, "get_or_create_collection", None)
        if not callable(get_collection) and not callable(get_or_create_collection):
            return None

        try:
            if callable(get_collection):
                collection = get_collection(collection_name)
            elif callable(get_or_create_collection):
                collection = get_or_create_collection(collection_name)
            else:
                return None
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            if not callable(get_or_create_collection):
                logger.debug(
                    "Unable to inspect vector collection '{}' for scoped embedding metadata: {}",
                    collection_name,
                    exc,
                )
                return None
            try:
                # Fresh Chroma clients can momentarily miss a just-written collection;
                # fall back to the same access path used by vector search.
                collection = get_or_create_collection(collection_name)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as fallback_exc:
                logger.debug(
                    "Unable to inspect vector collection '{}' for scoped embedding metadata: {}",
                    collection_name,
                    fallback_exc,
                )
                return None

        get_items = getattr(collection, "get", None)
        if not callable(get_items):
            return None

        try:
            metadata_payload = get_items(
                where=scoped_filter,
                include=["metadatas"],
                limit=5,
            )
        except TypeError:
            try:
                metadata_payload = get_items(include=["metadatas"], limit=5)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug(
                    "Vector metadata lookup fallback failed for collection '{}': {}",
                    collection_name,
                    exc,
                )
                return None
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug(
                "Vector metadata lookup failed for collection '{}': {}",
                collection_name,
                exc,
            )
            return None

        metadatas = self._extract_collection_metadatas(metadata_payload)
        if scoped_media_ids:
            metadatas = [
                item
                for item in metadatas
                if str(item.get("media_id")) in scoped_media_ids
            ]

        model_overrides: set[str] = set()
        for metadata in metadatas:
            embedding_model = str(metadata.get("embedding_model") or "").strip()
            if not embedding_model:
                continue
            embedding_provider = str(metadata.get("embedding_provider") or "").strip()
            if embedding_provider:
                model_overrides.add(f"{embedding_provider}:{embedding_model}")
            else:
                model_overrides.add(embedding_model)

        if len(model_overrides) == 1:
            return next(iter(model_overrides))

        if len(model_overrides) > 1:
            logger.warning(
                "Scoped media selection spans multiple embedding models in collection '{}': {}",
                collection_name,
                sorted(model_overrides),
            )
        return None

    async def _retrieve_vector(
        self,
        query: str,
        media_type: Optional[str] = None,
        **kwargs
    ) -> list[Document]:
        """
        Retrieve documents using vector search.

        Args:
            query: Search query text
            media_type: Optional media type filter

        Returns:
            List of documents from vector search
        """
        vector_store = self.vector_store
        if vector_store is None:
            logger.warning("Vector store not initialized, falling back to FTS")
            return await self._retrieve_fts(query, media_type, **kwargs)

        def _finalize_docs(documents: list[Document]) -> list[Document]:
            for doc in documents:
                _apply_location_metadata(doc)
            return documents

        try:
            # Allow callers to provide a precomputed query vector (e.g., HyDE)
            provided_vector = kwargs.get("query_vector")
            # Initialize vector store if needed
            if not vector_store._initialized:
                await vector_store.initialize()

            # Search collection selection; support multi-search via wildcard/list namespace
            index_namespace = kwargs.get("index_namespace")
            multi_namespace: Optional[list[str]] = None
            collection_name: Optional[str] = None
            if index_namespace:
                # If a list/tuple of patterns provided, use multi_search
                if isinstance(index_namespace, (list, tuple)):
                    multi_namespace = [str(x) for x in index_namespace]
                # If a single string contains a wildcard, treat as pattern
                elif isinstance(index_namespace, str) and ("*" in index_namespace or "?" in index_namespace):
                    multi_namespace = [index_namespace]
                else:
                    # Use provided namespace directly (already includes user prefix if desired)
                    collection_name = str(index_namespace)
            else:
                # Default: user-specific media collection
                collection_name = f"user_{self.user_id}_media_embeddings"

            # Get embedding for query (or use provided)
            if provided_vector is not None:
                query_vector = provided_vector
            else:
                try:
                    # Import only when we actually need to generate embeddings to avoid
                    # side effects (e.g., duplicate Prometheus collectors in tests)
                    from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
                        create_embeddings_batch,
                        get_embedding_config,
                    )
                    user_app_config = get_embedding_config()
                    model_id_override = self._resolve_scoped_query_embedding_override(
                        collection_name=collection_name,
                        allowed_media_ids=kwargs.get("allowed_media_ids"),
                    )
                    embeddings = await asyncio.get_event_loop().run_in_executor(
                        None,
                        create_embeddings_batch,
                        [query],  # texts
                        user_app_config,
                        model_id_override,
                    )

                    if not embeddings or not embeddings[0]:
                        logger.error("Failed to generate query embedding")
                        return []

                    query_vector = embeddings[0]
                    if hasattr(query_vector, 'tolist'):
                        query_vector = query_vector.tolist()
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                    logger.error(f"Failed to generate query embedding: {e}")
                    return []

            # Build filter for vector search
            base_filter: dict[str, Any] = {}
            if media_type:
                base_filter["media_type"] = media_type

            # Optional rich JSONB filter to be combined with base_filter
            metadata_filter = kwargs.get("metadata_filter")
            if isinstance(metadata_filter, dict) and metadata_filter:
                if base_filter:
                    # Combine via AND to preserve both constraints
                    base_filter = {"$and": [base_filter, metadata_filter]}
                else:
                    base_filter = metadata_filter
            base_filter = self._merge_vector_filters(
                base_filter if base_filter else None,
                self._build_allowed_media_vector_filter(kwargs.get("allowed_media_ids")),
            )

            # HYDE-aware retrieval and merge
            try:
                from tldw_Server_API.app.core.config import settings as _settings  # late import
                hyde_enabled = bool(_settings.get("HYDE_ENABLED", False))
                hyde_only_if_needed = bool(_settings.get("HYDE_ONLY_IF_NEEDED", True))
                hyde_score_floor = float(_settings.get("HYDE_SCORE_FLOOR", 0.30) or 0.30)
                hyde_k_frac = float(_settings.get("HYDE_K_FRACTION", 0.5) or 0.5)
                hyde_weight = float(_settings.get("HYDE_WEIGHT_QUESTION_MATCH", 0.05) or 0.05)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                hyde_enabled = False
                hyde_only_if_needed = True
                hyde_score_floor = 0.30
                hyde_k_frac = 0.5
                hyde_weight = 0.05

            k = int(self.config.max_results or 10)

            async def _search_with_filter(kind_filter: Optional[str], k_: int):
                # Merge kind filter with base_filter
                f: Optional[dict[str, Any]]
                if kind_filter:
                    f = {"$and": [base_filter, {"kind": kind_filter}]} if base_filter else {"kind": kind_filter}
                else:
                    f = base_filter if base_filter else None

                try:
                    # Multi-namespace search when patterns provided
                    if multi_namespace:
                        return await vector_store.multi_search(
                            collection_patterns=multi_namespace,
                            query_vector=query_vector,
                            k=k_,
                            filter=f,
                        )
                    # Single collection search
                    if collection_name is None:
                        return []
                    return await vector_store.search(
                        collection_name=collection_name,
                        query_vector=query_vector,
                        k=k_,
                        filter=f,
                        include_metadata=True,
                    )
                except (AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    logger.debug(f"Vector search with filter {f} failed, reason={exc}")
                    return []

            # 1) Baseline chunk search (prefer kind='chunk', fallback to no kind filter)
            base_results = await _search_with_filter("chunk", k)
            if not base_results:
                base_results = await _search_with_filter(None, k)

            # Optional early exit
            max_base = max((r.score for r in base_results), default=0.0)
            if not hyde_enabled or (hyde_only_if_needed and len(base_results) >= k and max_base >= hyde_score_floor):
                # Convert and return baseline
                documents: list[Document] = []
                allowed_media_ids = kwargs.get("allowed_media_ids")
                allowed_set = {int(x) for x in allowed_media_ids} if allowed_media_ids else None
                for result in base_results:
                    doc_media_id = result.metadata.get("media_id", result.id)
                    try:
                        doc_media_id_int = int(str(doc_media_id))
                    except (TypeError, ValueError):
                        doc_media_id_int = None
                    if allowed_set is not None and (doc_media_id_int is None or doc_media_id_int not in allowed_set):
                        continue
                    documents.append(
                        Document(
                            id=result.metadata.get("media_id", result.id),
                            content=result.content,
                            metadata=result.metadata,
                            score=result.score,
                            source=DataSource.MEDIA_DB,
                        )
                    )
                logger.debug(f"Retrieved {len(documents)} documents from vector search (baseline)")
                return _finalize_docs(documents)

            # 2) HYDE search on question vectors
            k_hyde = max(1, int(k * hyde_k_frac))
            hyde_results = await _search_with_filter("hyde_q", k_hyde)

            # 3) Merge - two modes: media-level (default) and optional chunk-level (by parent_chunk_id)
            try:
                dedupe_by_parent = bool(_settings.get("HYDE_DEDUPE_BY_PARENT", False))
            except (AttributeError, TypeError, ValueError):
                dedupe_by_parent = False

            if not dedupe_by_parent:
                # Media-level merge (default)
                best_score: dict[str, float] = {}
                doc_map: dict[str, Document] = {}

                def _maybe_add_media(result_obj):
                    # Convert VectorSearchResult to Document (id=media_id preferred)
                    d_id = result_obj.metadata.get("media_id", result_obj.id)
                    d = Document(
                        id=d_id,
                        content=result_obj.content,
                        metadata=result_obj.metadata,
                        score=result_obj.score,
                        source=DataSource.MEDIA_DB,
                    )
                    if d_id not in doc_map:
                        doc_map[d_id] = d
                    prev = best_score.get(d_id, 0.0)
                    if d.score > prev:
                        best_score[d_id] = d.score
                        doc_map[d_id].score = d.score

                # Add baseline first
                for r in base_results:
                    _maybe_add_media(r)

                # Add/adjust HYDE
                for r in hyde_results:
                    adj = min(1.0, r.score + hyde_weight)
                    d_id = r.metadata.get("media_id", r.id)
                    prev = best_score.get(d_id, 0.0)
                    if adj > prev:
                        best_score[d_id] = adj
                        if d_id in doc_map:
                            doc_map[d_id].score = adj
                        else:
                            doc_map[d_id] = Document(
                                id=d_id,
                                content=r.content,
                                metadata=r.metadata,
                                score=adj,
                                source=DataSource.MEDIA_DB,
                            )

                allowed_media_ids = kwargs.get("allowed_media_ids")
                allowed_set = {int(x) for x in allowed_media_ids} if allowed_media_ids else None
                merged = list(doc_map.values())
                if allowed_set is not None:
                    tmp = []
                    for d in merged:
                        try:
                            mid = int(str(d.metadata.get("media_id", d.id)))
                        except (TypeError, ValueError):
                            mid = None
                        if mid is not None and mid in allowed_set:
                            tmp.append(d)
                    merged = tmp
                merged.sort(key=lambda d: d.score, reverse=True)
                documents = merged[:k]
                logger.debug(f"Retrieved {len(documents)} documents from vector search (HYDE merged, media-level)")
                return _finalize_docs(documents)

            # Chunk-level merge (by parent_chunk_id)
            # Build best-per-parent maps for base and HYDE
            base_by_parent: dict[str, tuple[float, Any]] = {}
            for r in base_results:
                parent = str(r.metadata.get("parent_chunk_id") or r.id)
                sc = float(r.score)
                if parent not in base_by_parent or sc > base_by_parent[parent][0]:
                    base_by_parent[parent] = (sc, r)

            hyde_by_parent: dict[str, tuple[float, Any]] = {}
            for r in hyde_results:
                parent = str(r.metadata.get("parent_chunk_id") or r.id)
                sc = min(1.0, float(r.score) + hyde_weight)
                if parent not in hyde_by_parent or sc > hyde_by_parent[parent][0]:
                    hyde_by_parent[parent] = (sc, r)

            # Combine per parent key
            chunk_keys = set(base_by_parent.keys()) | set(hyde_by_parent.keys())
            chunk_docs: list[Document] = []
            for ck in chunk_keys:
                base_tuple = base_by_parent.get(ck)
                hyde_tuple = hyde_by_parent.get(ck)
                base_sc = base_tuple[0] if base_tuple else 0.0
                hyde_sc = hyde_tuple[0] if hyde_tuple else 0.0
                best_sc = max(base_sc, hyde_sc)
                # Prefer base result's metadata/content when present; else use HYDE
                src = base_tuple[1] if base_tuple else (hyde_tuple[1] if hyde_tuple else None)
                if not src:
                    continue
                d_id = src.metadata.get("media_id", src.id)
                d = Document(
                    id=d_id,
                    content=src.content,
                    metadata=src.metadata,
                    score=best_sc,
                    source=DataSource.MEDIA_DB,
                )
                chunk_docs.append(d)

            # Apply allowed_media_ids filter
            allowed_media_ids = kwargs.get("allowed_media_ids")
            allowed_set = {int(x) for x in allowed_media_ids} if allowed_media_ids else None
            if allowed_set is not None:
                filtered: list[Document] = []
                for d in chunk_docs:
                    try:
                        mid = int(str(d.metadata.get("media_id", d.id)))
                    except (TypeError, ValueError):
                        mid = None
                    if mid is not None and mid in allowed_set:
                        filtered.append(d)
                chunk_docs = filtered

            chunk_docs.sort(key=lambda d: d.score, reverse=True)
            documents = chunk_docs[:k]
            logger.debug(f"Retrieved {len(documents)} documents from vector search (HYDE merged, chunk-level)")
            return _finalize_docs(documents)

        except (AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError) as e:
            logger.error(f"Vector search failed: {e}")
            # Fallback to FTS
            return await self._retrieve_fts(query, media_type, **kwargs)

    async def retrieve_hybrid(
        self,
        query: str,
        media_type: Optional[str] = None,
        alpha: float = 0.7,
        **kwargs
    ) -> list[Document]:
        """
        Retrieve documents using hybrid search (FTS + Vector).

        Args:
            query: Search query
            media_type: Optional media type filter
            alpha: Weight for vector search (0=FTS only, 1=Vector only)

        Returns:
            Merged and re-ranked documents
        """
        # Perform both searches in parallel
        fts_task = self._retrieve_fts(query, media_type, **kwargs)
        vector_task = self._retrieve_vector(query, media_type, **kwargs)

        fts_docs, vector_docs = await asyncio.gather(fts_task, vector_task)

        # Merge using reciprocal rank fusion
        return self._reciprocal_rank_fusion(fts_docs, vector_docs, alpha)

    def _reciprocal_rank_fusion(
        self,
        fts_docs: list[Document],
        vector_docs: list[Document],
        alpha: float = 0.7,
        k: int = 60
    ) -> list[Document]:
        """
        Merge FTS and vector results using reciprocal rank fusion.

        Args:
            fts_docs: Documents from FTS search
            vector_docs: Documents from vector search
            alpha: Weight for vector search (0=FTS only, 1=Vector only)
            k: Constant for RRF (typically 60)

        Returns:
            Merged and re-ranked documents
        """
        # Create score dictionaries
        fts_scores = {}
        vector_scores = {}
        doc_map = {}

        # Calculate RRF scores for FTS results
        for rank, doc in enumerate(fts_docs):
            doc_id = doc.id
            fts_scores[doc_id] = 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc

        # Calculate RRF scores for vector results
        for rank, doc in enumerate(vector_docs):
            doc_id = doc.id
            vector_scores[doc_id] = 1.0 / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        # Combine scores with weighting
        final_scores = {}
        all_doc_ids = set(fts_scores.keys()) | set(vector_scores.keys())

        for doc_id in all_doc_ids:
            fts_score = fts_scores.get(doc_id, 0)
            vector_score = vector_scores.get(doc_id, 0)
            # Weighted combination
            final_scores[doc_id] = (1 - alpha) * fts_score + alpha * vector_score

        # Sort by final score
        sorted_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)

        # Create final document list
        merged_docs = []
        for doc_id in sorted_ids[:self.config.max_results]:
            doc = doc_map[doc_id]
            doc.score = final_scores[doc_id]
            merged_docs.append(doc)

        logger.debug(f"Hybrid search merged {len(merged_docs)} documents")
        return merged_docs

    async def get_metadata(self, doc_id: str) -> dict[str, Any]:
        """Get full metadata for a media item."""
        aggregator = "GROUP_CONCAT(t.name)"
        media_adapter = getattr(self, 'media_db', None)
        if media_adapter is not None and getattr(media_adapter, 'backend_type', None) == BackendType.POSTGRESQL:
            aggregator = "STRING_AGG(t.name, ',')"
        metadata_sql_template = """
            SELECT
                m.*,
                {aggregator} as tags,
                COUNT(DISTINCT ma.id) as analysis_count
            FROM media m
            LEFT JOIN media_tags mt ON m.id = mt.media_id
            LEFT JOIN tags t ON mt.tag_id = t.id
            LEFT JOIN media_analysis ma ON m.id = ma.media_id
            WHERE m.id = ?
            GROUP BY m.id
        """
        sql = metadata_sql_template.format_map(locals())  # nosec B608

        results = self._execute_query(sql, (doc_id,))

        if results:
            row = results[0]
            return {
                "id": row["id"],
                "title": row["title"],
                "media_type": row["media_type"],
                "url": row["url"],
                "tags": row["tags"].split(",") if row["tags"] else [],
                "analysis_count": row["analysis_count"],
                "created_at": row["created_at"]
            }

        return {}

    def _build_fts_query(self, query: str) -> str:
        """Build FTS5 query with proper escaping and hyphen/unicode handling.

        - If multiple tokens or hyphens present, use a quoted phrase to preserve order.
        - Otherwise, apply prefix match (token*).
        - Escape embedded quotes for FTS5.
        """
        text = (query or "").strip()
        if not text:
            return "*"
        # Normalize quotes
        safe = text.replace('"', '""')
        # Heuristic: phrase if contains whitespace, hyphens, or parentheses/quotes
        if any(ch in safe for ch in [" ", "-", "(", ")", "'", "\u2013", "\u2014"]):
            return f'"{safe}"'
        # Single token: prefix
        return f"{safe}*"


class NotesDBRetriever(BaseRetriever):
    """Retriever for notes database."""

    def __init__(
        self,
        db_path: Optional[str],
        config: Optional[RetrievalConfig] = None,
        *,
        chacha_db: Optional['CharactersRAGDB'] = None
    ) -> None:
        super().__init__(db_path, config, db_adapter=chacha_db)
        self.chacha_db = chacha_db

    async def retrieve(
        self,
        query: str,
        notebook_id: Optional[int] = None,
        **kwargs
    ) -> list[Document]:
        """Retrieve from notes database."""
        if self.chacha_db is not None and not self.config.tags_filter:
            docs = self._retrieve_via_chacha(query, notebook_id)
            # Optional restriction to specific note IDs
            allowed_note_ids = kwargs.get("allowed_note_ids")
            if allowed_note_ids and isinstance(allowed_note_ids, (list, tuple)):
                allowed_set = {str(x) for x in allowed_note_ids}
                docs = [d for d in docs if str(d.id).replace("note_", "") in allowed_set]
            return docs

        documents = []

        # Build SQL query compatible with ChaChaNotes schema (no notebooks/tags tables)
        sql = """
            SELECT
                n.id,
                n.title,
                n.content,
                n.created_at,
                n.last_modified AS updated_at
            FROM notes n
            WHERE n.deleted = 0 AND (n.title LIKE ? OR n.content LIKE ?)
        """
        params: list[Any] = [f"%{query}%", f"%{query}%"]

        # Optional restriction to specific note IDs (e.g., from access control)
        allowed_note_ids = kwargs.get("allowed_note_ids")
        if allowed_note_ids and isinstance(allowed_note_ids, (list, tuple)):
            placeholders = ",".join(["?"] * len(allowed_note_ids))
            sql += f" AND n.id IN ({placeholders})"
            params.extend(list(allowed_note_ids))

        # Order and limit
        sql += " ORDER BY n.last_modified DESC LIMIT ?"
        params.append(self.config.max_results)

        # Execute query
        results = self._execute_query(sql, tuple(params))

        # Convert to documents
        for row in results:
            # Calculate simple relevance score
            title_match = query.lower() in row["title"].lower()
            content_match = query.lower() in row["content"].lower()
            score = (1.0 if title_match else 0.0) + (0.5 if content_match else 0.0)

            doc = Document(
                id=f"note_{row['id']}",
                content=f"# {row['title']}\n\n{row['content']}",
                source=DataSource.NOTES,  # Add required source parameter
                metadata={
                    "title": row["title"],
                    "notebook": None,
                    "notebook_id": None,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "source": "notes_db"
                },
                score=score
            )
            documents.append(doc)

        # Sort by score
        documents.sort(key=lambda x: x.score, reverse=True)

        logger.debug(f"Retrieved {len(documents)} documents from Notes_DB")

        return documents

    def _retrieve_via_chacha(self, query: str, notebook_id: Optional[int]) -> list[Document]:
        if self.chacha_db is None:
            return []
        try:
            results = self.chacha_db.search_notes(query, limit=int(self.config.max_results))
        except (AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.error(f"ChaCha notes search failed: {exc}")
            return []
        documents: list[Document] = []
        # Normalize backend-provided ranks (if present) to [0,1]
        ranks = []
        for r in results:
            rv = r.get('rank')
            if rv is None:
                rv = r.get('bm25_score')
            try:
                ranks.append(float(rv) if rv is not None else None)
            except (TypeError, ValueError):
                ranks.append(None)
        norm_map = {}
        if any(v is not None for v in ranks):
            vals = [v for v in ranks if v is not None]
            if getattr(self.chacha_db, 'backend_type', None) == BackendType.POSTGRESQL:
                scaled = _normalize_scores(vals, method="minmax")
            else:
                scaled = _normalize_scores([-v for v in vals], method="minmax")
            it = iter(scaled)
            for idx, v in enumerate(ranks):
                if v is not None:
                    norm_map[idx] = float(next(it))
        min_score = float(self.config.min_score or 0.0)
        for idx, row in enumerate(results):
            if notebook_id and row.get('notebook_id') != notebook_id:
                continue
            score_val = norm_map.get(idx, 0.75)
            if score_val < min_score:
                continue
            metadata = {}
            if self.config.include_metadata:
                metadata = {
                    'title': row.get('title'),
                    'notebook': row.get('notebook_name'),
                    'notebook_id': row.get('notebook_id'),
                    'created_at': row.get('created_at'),
                    'updated_at': row.get('updated_at'),
                    'source': 'notes_db',
                }
            documents.append(
                Document(
                    id=f"note_{row.get('id')}",
                    content=f"# {row.get('title')}\n\n{row.get('content', '')}",
                    source=DataSource.NOTES,
                    metadata=metadata,
                    score=float(score_val),
                )
            )
        documents.sort(key=lambda x: getattr(x, 'score', 0.0), reverse=True)
        return documents

    async def get_metadata(self, doc_id: str) -> dict[str, Any]:
        """Get metadata for a note."""
        # Extract numeric ID
        note_id = doc_id.replace("note_", "")

        aggregator = "GROUP_CONCAT(t.name)"
        if self.chacha_db is not None and getattr(self.chacha_db, 'backend_type', None) == BackendType.POSTGRESQL:
            aggregator = "STRING_AGG(t.name, ',')"
        metadata_sql_template = """
            SELECT
                n.*,
                nb.name as notebook_name,
                {aggregator} as tags
            FROM notes n
            LEFT JOIN notebooks nb ON n.notebook_id = nb.id
            LEFT JOIN note_tags nt ON n.id = nt.note_id
            LEFT JOIN tags t ON nt.tag_id = t.id
            WHERE n.id = ?
            GROUP BY n.id
        """
        sql = metadata_sql_template.format_map(locals())  # nosec B608

        results = self._execute_query(sql, (note_id,))

        if results:
            row = results[0]
            return {
                "id": row["id"],
                "title": row["title"],
                "notebook": row["notebook_name"],
                "tags": row["tags"].split(",") if row["tags"] else [],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }

        return {}


class KanbanDBRetriever(BaseRetriever):
    """Retriever for Kanban cards."""

    _FTS_WEIGHT = _get_float_env("KANBAN_SEARCH_FTS_WEIGHT", 0.6)
    _VECTOR_WEIGHT = _get_float_env("KANBAN_SEARCH_VECTOR_WEIGHT", 0.4)
    _VECTOR_ONLY_WEIGHT = _get_float_env("KANBAN_SEARCH_VECTOR_ONLY_WEIGHT", 0.3)

    def __init__(
        self,
        db_path: Optional[str],
        config: Optional[RetrievalConfig] = None,
        *,
        user_id: str = "0",
        kanban_db: Optional[KanbanDB] = None,
    ) -> None:
        super().__init__(db_path, config, db_adapter=kanban_db)
        self._owns_db = False
        self.kanban_db = kanban_db
        self.user_id = self._resolve_user_id(user_id)
        if self.kanban_db is None and self.db_path is not None:
            self.kanban_db = KanbanDB(db_path=str(self.db_path), user_id=self.user_id)
            self._owns_db = True

    @staticmethod
    def _resolve_user_id(user_id: Optional[str]) -> str:
        raw = str(user_id).strip() if user_id is not None else ""
        return raw or "0"

    def _get_db(self) -> KanbanDB:
        if self.kanban_db is None:
            if self.db_path is None:
                raise MissingDatabasePathError()
            self.kanban_db = KanbanDB(db_path=str(self.db_path), user_id=self.user_id)
            self._owns_db = True
        return self.kanban_db

    async def retrieve(
        self,
        query: str,
        *,
        board_id: Optional[int] = None,
        label_ids: Optional[list[int]] = None,
        priority: Optional[str] = None,
        include_archived: bool = False,
        **kwargs: Any,
    ) -> list[Document]:
        if not query or not str(query).strip():
            return []
        db = self._get_db()
        limit = int(getattr(self.config, "max_results", 20) or 20)

        if getattr(self.config, "use_vector", False) and getattr(self.config, "use_fts", True):
            cards = self._hybrid_search(
                db,
                query=query,
                board_id=board_id,
                label_ids=label_ids,
                priority=priority,
                include_archived=include_archived,
                limit=limit,
            )
        elif getattr(self.config, "use_vector", False):
            cards = self._vector_search(
                db,
                query=query,
                board_id=board_id,
                label_ids=label_ids,
                priority=priority,
                include_archived=include_archived,
                limit=limit,
            )
        else:
            cards = self._fts_search(
                db,
                query=query,
                board_id=board_id,
                label_ids=label_ids,
                priority=priority,
                include_archived=include_archived,
                limit=limit,
            )

        return self._cards_to_documents(cards, min_score=float(self.config.min_score or 0.0))

    def _fts_search(
        self,
        db: KanbanDB,
        *,
        query: str,
        board_id: Optional[int],
        label_ids: Optional[list[int]],
        priority: Optional[str],
        include_archived: bool,
        limit: int,
    ) -> list[dict[str, Any]]:
        cards, _ = db.search_cards(
            query=query,
            board_id=board_id,
            label_ids=label_ids,
            priority=priority,
            include_archived=include_archived,
            limit=limit,
            offset=0,
        )
        total = max(len(cards), 1)
        for idx, card in enumerate(cards):
            card["relevance_score"] = 1.0 - (idx / total)
        return cards

    def _vector_search(
        self,
        db: KanbanDB,
        *,
        query: str,
        board_id: Optional[int],
        label_ids: Optional[list[int]],
        priority: Optional[str],
        include_archived: bool,
        limit: int,
    ) -> list[dict[str, Any]]:
        vector_search = db.get_vector_search()
        if vector_search is None or not getattr(vector_search, "available", False):
            return self._fts_search(
                db,
                query=query,
                board_id=board_id,
                label_ids=label_ids,
                priority=priority,
                include_archived=include_archived,
                limit=limit,
            )
        results = vector_search.search(
            query=query,
            board_id=board_id,
            priority=priority,
            limit=limit,
        )
        if not results:
            return []
        card_ids = [r.get("card_id") for r in results if r.get("card_id")]
        score_map = {r["card_id"]: r.get("relevance_score", 0.0) for r in results if r.get("card_id")}
        cards = db.get_cards_by_ids(
            card_ids=card_ids,
            include_deleted=False,
            include_archived=include_archived,
        )
        filtered = self._filter_by_labels(cards, label_ids)
        for card in filtered:
            card["relevance_score"] = score_map.get(card["id"], 0.0)
        filtered.sort(key=lambda c: c.get("relevance_score", 0.0), reverse=True)
        return filtered[:limit]

    def _hybrid_search(
        self,
        db: KanbanDB,
        *,
        query: str,
        board_id: Optional[int],
        label_ids: Optional[list[int]],
        priority: Optional[str],
        include_archived: bool,
        limit: int,
    ) -> list[dict[str, Any]]:
        fts_cards = self._fts_search(
            db,
            query=query,
            board_id=board_id,
            label_ids=label_ids,
            priority=priority,
            include_archived=include_archived,
            limit=limit * 2,
        )
        vector_search = db.get_vector_search()
        if vector_search is None or not getattr(vector_search, "available", False):
            return fts_cards[:limit]
        vector_results = vector_search.search(
            query=query,
            board_id=board_id,
            priority=priority,
            limit=limit * 2,
        )
        if not vector_results:
            return fts_cards[:limit]
        vector_scores = {
            r["card_id"]: r.get("relevance_score", 0.0)
            for r in vector_results
            if r.get("card_id")
        }
        combined: list[dict[str, Any]] = []
        seen: set[int] = set()
        for _idx, card in enumerate(fts_cards):
            card_id = card["id"]
            if card_id in seen:
                continue
            seen.add(card_id)
            fts_score = card.get("relevance_score", 0.0)
            vector_score = vector_scores.get(card_id, 0.0)
            card["relevance_score"] = (self._FTS_WEIGHT * fts_score) + (self._VECTOR_WEIGHT * vector_score)
            combined.append(card)
        extra_ids = [cid for cid in vector_scores if cid not in seen]
        if extra_ids:
            extra_cards = db.get_cards_by_ids(
                card_ids=extra_ids,
                include_deleted=False,
                include_archived=include_archived,
            )
            extra_cards = self._filter_by_labels(extra_cards, label_ids)
            for card in extra_cards:
                card["relevance_score"] = self._VECTOR_ONLY_WEIGHT * vector_scores.get(card["id"], 0.0)
            combined.extend(extra_cards)
        combined.sort(key=lambda c: c.get("relevance_score", 0.0), reverse=True)
        return combined[:limit]

    @staticmethod
    def _filter_by_labels(cards: list[dict[str, Any]], label_ids: Optional[list[int]]) -> list[dict[str, Any]]:
        if not label_ids:
            return list(cards)
        required = set(label_ids)
        filtered = []
        for card in cards:
            card_label_ids = {label.get("id") for label in card.get("labels", []) if label.get("id") is not None}
            if required.issubset(card_label_ids):
                filtered.append(card)
        return filtered

    def _cards_to_documents(self, cards: list[dict[str, Any]], *, min_score: float) -> list[Document]:
        documents: list[Document] = []
        for card in cards:
            score = float(card.get("relevance_score") or 0.0)
            if score < min_score:
                continue
            content_parts = [card.get("title") or ""]
            description = card.get("description") or ""
            if description:
                content_parts.append(description)
            labels = card.get("labels") or []
            label_names = [label.get("name") for label in labels if label.get("name")]
            if label_names:
                content_parts.append("Labels: " + ", ".join(label_names))
            content = "\n\n".join([p for p in content_parts if p]).strip()
            documents.append(
                Document(
                    id=f"kanban_card_{card.get('id')}",
                    content=content,
                    source=DataSource.KANBAN,
                    metadata={
                        "card_id": card.get("id"),
                        "card_uuid": card.get("uuid"),
                        "board_id": card.get("board_id"),
                        "board_name": card.get("board_name"),
                        "list_id": card.get("list_id"),
                        "list_name": card.get("list_name"),
                        "due_date": card.get("due_date"),
                        "priority": card.get("priority"),
                        "created_at": card.get("created_at"),
                        "updated_at": card.get("updated_at"),
                        "source": "kanban",
                    },
                    score=score,
                )
            )
        documents.sort(key=lambda d: getattr(d, "score", 0.0), reverse=True)
        return documents

    async def get_metadata(self, doc_id: str) -> dict[str, Any]:
        """Get metadata for a Kanban card document."""
        if not doc_id:
            return {}

        card_id_str = doc_id
        if doc_id.startswith("kanban_card_"):
            card_id_str = doc_id.replace("kanban_card_", "", 1)

        try:
            card_id = int(card_id_str)
        except (TypeError, ValueError):
            logger.debug(f"KanbanDBRetriever.get_metadata received non-numeric doc_id: {doc_id}")
            return {}

        try:
            db = self._get_db()
            card = db.get_card_with_details(card_id, include_deleted=False) or db.get_card(
                card_id, include_deleted=False
            )
            if not card:
                return {}
            metadata = dict(card)
            metadata.setdefault("source", "kanban")
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug(f"KanbanDBRetriever metadata lookup failed for {doc_id}: {exc}")
        else:
            return metadata
        return {}

    def close(self) -> None:
        if self._owns_db and self.kanban_db is not None:
            try:
                self.kanban_db.close()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug(f"KanbanDBRetriever close error: {exc}")
            finally:
                self.kanban_db = None
                self._owns_db = False


class PromptsDBRetriever(BaseRetriever):
    """Retriever for prompts database."""

    def __init__(
        self,
        db_path: Optional[str],
        config: Optional[RetrievalConfig] = None,
        *,
        chacha_db: Optional['CharactersRAGDB'] = None,
        prompts_db: Optional[Any] = None,
    ) -> None:
        super().__init__(db_path, config, db_adapter=prompts_db or chacha_db)
        self.prompts_db = prompts_db or chacha_db

    def _document_from_prompt_row(
        self,
        row: dict[str, Any],
        query_terms: list[str],
    ) -> Document:
        name = str(row.get("name") or "(Untitled prompt)")
        details = str(row.get("details") or "")
        system_prompt = str(row.get("system_prompt") or "")
        user_prompt = str(row.get("user_prompt") or "")
        searchable_parts = [name, details, system_prompt, user_prompt]
        matched_terms = sum(
            1
            for term in query_terms
            if any(term in part.lower() for part in searchable_parts if part)
        )
        usage_count = _coerce_int(row.get("usage_count")) or 0
        score = min(
            1.0,
            (matched_terms / max(len(query_terms), 1)) + min(usage_count / 100.0, 0.2),
        )

        prompt_id = row.get("id")
        content = (
            f"**{name}**\n\n"
            f"{details}\n\n"
            f"System:\n{system_prompt}\n\n"
            f"User:\n{user_prompt}"
        ).strip()
        return Document(
            id=f"prompt_{prompt_id}",
            content=content,
            source=DataSource.PROMPTS,
            metadata={
                "prompt_id": prompt_id,
                "name": name,
                "author": row.get("author"),
                "uuid": row.get("uuid"),
                "version": row.get("version"),
                "usage_count": usage_count,
                "last_modified": row.get("last_modified"),
                "source": "prompts",
            },
            score=float(score),
        )

    async def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """
        Retrieve from prompts database.

        Args:
            query: Search query
            category: Optional category filter

        Returns:
            List of retrieved documents
        """
        search_prompts = getattr(self.prompts_db, "search_prompts", None)
        max_results = int(self.config.max_results)
        query_terms = [term for term in re.findall(r"[A-Za-z0-9']+", query.lower()) if term]
        if callable(search_prompts):
            try:
                results, _total = await asyncio.to_thread(
                    search_prompts,
                    search_query=query,
                    search_fields=["name", "details", "system_prompt", "user_prompt"],
                    page=1,
                    results_per_page=max_results,
                    include_deleted=False,
                )
                documents = [
                    self._document_from_prompt_row(dict(row), query_terms)
                    for row in results
                ]
                documents.sort(key=lambda doc: doc.score, reverse=True)
                return documents
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as exc:
                logger.debug(f"PromptsDatabase search_prompts failed; falling back to SQL search: {exc}")

        # Build SQL query
        sql = """
            SELECT
                p.id,
                p.name,
                p.author,
                p.details,
                p.system_prompt,
                p.user_prompt,
                p.uuid,
                p.last_modified,
                p.version,
                p.usage_count
            FROM Prompts p
            WHERE p.deleted = 0
              AND (
                p.name LIKE ?
                OR p.details LIKE ?
                OR p.system_prompt LIKE ?
                OR p.user_prompt LIKE ?
              )
        """

        params: list[Any] = [f"%{query}%"] * 4

        sql += " ORDER BY p.usage_count DESC, p.last_modified DESC LIMIT ?"
        params.append(max_results)

        # Execute query
        results = await self._execute_query_async(sql, tuple(params))

        # Convert to documents
        documents = [self._document_from_prompt_row(row, query_terms) for row in results]

        # Sort by score
        documents.sort(key=lambda x: x.score, reverse=True)

        logger.debug(f"Retrieved {len(documents)} documents from Prompts_DB")

        return documents

    async def get_metadata(self, doc_id: str) -> dict[str, Any]:
        """Get metadata for a prompt."""
        prompt_id = doc_id.replace("prompt_", "")

        sql = "SELECT * FROM Prompts WHERE id = ?"
        results = self._execute_query(sql, (prompt_id,))

        if results:
            row = results[0]
            return dict(row)

        return {}


class ChatHistoryRetriever(BaseRetriever):
    """Retriever for chat conversation messages."""

    def __init__(
        self,
        db_path: Optional[str],
        config: Optional[RetrievalConfig] = None,
        *,
        chacha_db: Optional['CharactersRAGDB'] = None,
    ) -> None:
        super().__init__(db_path, config, db_adapter=chacha_db)
        self.chacha_db = chacha_db

    async def retrieve(self, query: str, **kwargs: Any) -> list[Document]:
        documents: list[Document] = []
        max_results = int(self.config.max_results)

        sql = """
            SELECT
                m.id,
                m.conversation_id,
                m.content,
                m.sender,
                m.timestamp,
                conv.character_id,
                cc.name AS character_name
            FROM messages m
            JOIN conversations conv ON m.conversation_id = conv.id
            LEFT JOIN character_cards cc ON conv.character_id = cc.id
            WHERE m.deleted = 0 AND m.content LIKE ?
            ORDER BY m.timestamp DESC
            LIMIT ?
        """
        rows = await self._execute_query_async(sql, (f"%{query}%", max_results))
        for row in rows:
            documents.append(
                Document(
                    id=f"chat_{row['id']}",
                    content=f"[{row.get('sender')}]: {row.get('content', '')}",
                    source=DataSource.CHAT_HISTORY,
                    metadata={
                        "message_id": row.get("id"),
                        "conversation_id": row.get("conversation_id"),
                        "sender": row.get("sender"),
                        "timestamp": row.get("timestamp"),
                        "character_id": row.get("character_id"),
                        "character_name": row.get("character_name"),
                        "type": "chat_message",
                        "source": "chats",
                    },
                    score=0.5,
                )
            )
        if documents:
            return documents

        if self.chacha_db is not None:
            try:
                msg_rows = await asyncio.to_thread(
                    self.chacha_db.search_messages_by_content,
                    query,
                    limit=max_results,
                )
                for row in msg_rows:
                    conv_id = row.get("conversation_id")

                    documents.append(
                        Document(
                            id=f"chat_{row['id']}",
                            content=f"{row.get('sender')}: {row.get('content', '')}",
                            source=DataSource.CHAT_HISTORY,
                            metadata={
                                "message_id": row.get("id"),
                                "conversation_id": conv_id,
                                "sender": row.get("sender"),
                                "timestamp": row.get("timestamp"),
                                "character_id": row.get("character_id"),
                                "character_name": row.get("character_name"),
                                "type": "chat_message",
                                "source": "chats",
                            },
                            score=0.5,
                        )
                    )
                return documents
            except (AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as exc:
                logger.debug(f"ChaCha chat search failed: {exc}")
        return documents

    async def get_metadata(self, doc_id: str) -> dict[str, Any]:
        chat_id = doc_id.replace("chat_", "")
        results = self._execute_query(
            """
            SELECT m.*, conv.character_id
            FROM messages m
            JOIN conversations conv ON m.conversation_id = conv.id
            WHERE m.id = ?
            """,
            (chat_id,),
        )
        return dict(results[0]) if results else {}


class WorldBooksRetriever(BaseRetriever):
    """Retriever for world books and lorebook entries."""

    async def retrieve(self, query: str, **kwargs: Any) -> list[Document]:
        max_results = int(self.config.max_results)
        sql = """
            SELECT
                wb.id AS world_book_id,
                wb.name AS world_book_name,
                wb.description AS world_book_description,
                e.id AS entry_id,
                e.keywords,
                e.content,
                e.priority,
                e.metadata,
                e.enabled
            FROM world_book_entries e
            JOIN world_books wb ON e.world_book_id = wb.id
            WHERE wb.deleted = 0
              AND wb.enabled = 1
              AND e.enabled = 1
              AND (
                wb.name LIKE ?
                OR wb.description LIKE ?
                OR e.keywords LIKE ?
                OR e.content LIKE ?
                OR e.metadata LIKE ?
              )
            ORDER BY e.priority DESC, e.id DESC
            LIMIT ?
        """
        rows = await self._execute_query_async(sql, (*([f"%{query}%"] * 5), max_results))
        documents: list[Document] = []
        query_lower = query.lower()
        for row in rows:
            keywords = row.get("keywords") or ""
            content = row.get("content") or ""
            name = row.get("world_book_name") or "(Untitled world book)"
            description = row.get("world_book_description") or ""
            matched_fields = sum(
                1
                for value in (name, description, keywords, content, row.get("metadata") or "")
                if query_lower in str(value).lower()
            )
            priority = _coerce_int(row.get("priority")) or 0
            documents.append(
                Document(
                    id=f"world_book_entry_{row['entry_id']}",
                    content=(
                        f"# {name}\n\n"
                        f"{description}\n\n"
                        f"Keywords: {keywords}\n\n"
                        f"{content}"
                    ).strip(),
                    source=DataSource.WORLD_BOOKS,
                    metadata={
                        "world_book_id": row.get("world_book_id"),
                        "world_book_name": name,
                        "entry_id": row.get("entry_id"),
                        "keywords": keywords,
                        "priority": priority,
                        "source": "world_books",
                    },
                    score=min(1.0, (matched_fields / 5.0) + min(priority / 100.0, 0.2)),
                )
            )
        documents.sort(key=lambda doc: doc.score, reverse=True)
        return documents

    async def get_metadata(self, doc_id: str) -> dict[str, Any]:
        entry_id = doc_id.replace("world_book_entry_", "")
        rows = self._execute_query(
            """
            SELECT e.*, wb.name AS world_book_name
            FROM world_book_entries e
            JOIN world_books wb ON e.world_book_id = wb.id
            WHERE e.id = ?
            """,
            (entry_id,),
        )
        return dict(rows[0]) if rows else {}


class ChatDictionariesRetriever(BaseRetriever):
    """Retriever for chat dictionaries and dictionary entries."""

    def __init__(
        self,
        db_path: Optional[str],
        config: Optional[RetrievalConfig] = None,
        *,
        chacha_db: Optional['CharactersRAGDB'] = None,
        db_adapter: Optional[Any] = None,
    ) -> None:
        super().__init__(db_path, config, db_adapter=db_adapter or chacha_db)
        self.chacha_db = chacha_db or db_adapter

    async def retrieve(self, query: str, **kwargs: Any) -> list[Document]:
        max_results = int(self.config.max_results)
        sql = """
            SELECT
                d.id AS dictionary_id,
                d.name AS dictionary_name,
                d.description,
                e.id AS entry_id,
                e.key,
                e.content,
                e.group_name
            FROM dictionary_entries e
            JOIN chat_dictionaries d ON e.dictionary_id = d.id
            WHERE d.deleted = 0
              AND d.is_active = 1
              AND e.enabled = 1
              AND (
                d.name LIKE ?
                OR d.description LIKE ?
                OR e.key LIKE ?
                OR e.content LIKE ?
                OR e.group_name LIKE ?
              )
            ORDER BY e.id ASC
            LIMIT ?
        """
        rows = await self._execute_query_async(sql, (*([f"%{query}%"] * 5), max_results))
        documents: list[Document] = []
        query_lower = query.lower()
        for row in rows:
            dictionary_name = row.get("dictionary_name") or "(Untitled dictionary)"
            key = row.get("key") or ""
            content = row.get("content") or ""
            fields = (
                dictionary_name,
                row.get("description") or "",
                key,
                content,
                row.get("group_name") or "",
            )
            matched_fields = sum(1 for value in fields if query_lower in str(value).lower())
            documents.append(
                Document(
                    id=f"dictionary_entry_{row['entry_id']}",
                    content=(
                        f"# {dictionary_name}\n\n"
                        f"Key: {key}\n\n"
                        f"{content}"
                    ).strip(),
                    source=DataSource.DICTIONARIES,
                    metadata={
                        "dictionary_id": row.get("dictionary_id"),
                        "dictionary_name": dictionary_name,
                        "entry_id": row.get("entry_id"),
                        "key": key,
                        "group_name": row.get("group_name"),
                        "source": "dictionaries",
                    },
                    score=matched_fields / 5.0,
                )
            )
        documents.sort(key=lambda doc: doc.score, reverse=True)
        return documents

    async def get_metadata(self, doc_id: str) -> dict[str, Any]:
        entry_id = doc_id.replace("dictionary_entry_", "")
        rows = self._execute_query(
            """
            SELECT e.*, d.name AS dictionary_name
            FROM dictionary_entries e
            JOIN chat_dictionaries d ON e.dictionary_id = d.id
            WHERE e.id = ?
            """,
            (entry_id,),
        )
        return dict(rows[0]) if rows else {}


class CharacterCardsRetriever(BaseRetriever):
    """Retriever for character cards and chats."""

    def __init__(
        self,
        db_path: Optional[str],
        config: Optional[RetrievalConfig] = None,
        *,
        chacha_db: Optional['CharactersRAGDB'] = None
    ) -> None:
        super().__init__(db_path, config, db_adapter=chacha_db)
        self.chacha_db = chacha_db

    async def retrieve(
        self,
        query: str,
        include_chats: bool = True,
        **kwargs
    ) -> list[Document]:
        """
        Retrieve from character cards and chats.

        Args:
            query: Search query
            include_chats: Whether to include chat messages

        Returns:
            List of retrieved documents
        """
        documents: list[Document] = []

        # Prefer backend-aware path via ChaCha DB (leverages Postgres FTS when available)
        if self.chacha_db is not None:
            try:
                limit_cards = max(1, self.config.max_results // 2)
                card_rows = self.chacha_db.search_character_cards(query, limit=limit_cards)
                # Normalize ranks if provided by backend (Postgres ts_rank or SQLite-derived)
                raw_ranks = []
                for r in card_rows:
                    rv = r.get("rank")
                    try:
                        raw_ranks.append(float(rv) if rv is not None else None)
                    except (TypeError, ValueError):
                        raw_ranks.append(None)
                norm_map = {}
                if any(v is not None for v in raw_ranks):
                    vals = [v for v in raw_ranks if v is not None]
                    if getattr(self.chacha_db, 'backend_type', None) == BackendType.POSTGRESQL:
                        scaled = _normalize_scores(vals, method="minmax")
                    else:
                        scaled = _normalize_scores([-v for v in vals], method="minmax")
                    it = iter(scaled)
                    for idx, v in enumerate(raw_ranks):
                        if v is not None:
                            norm_map[idx] = float(next(it))
                min_score = float(self.config.min_score or 0.0)
                for idx, row in enumerate(card_rows):
                    name = row.get("name") or "(Unnamed)"
                    description = row.get("description") or ""
                    personality = row.get("personality") or ""
                    scenario = row.get("scenario") or ""
                    first_message = row.get("first_message") or ""
                    score_val = norm_map.get(idx, 0.75)
                    if score_val < min_score:
                        continue

                    content = (
                        f"# {name}\n\n"
                        f"**Description:** {description}\n\n"
                        f"**Personality:** {personality}\n\n"
                        f"**Scenario:** {scenario}\n\n"
                        f"**First Message:** {first_message}"
                    )
                    doc = Document(
                        id=f"character_{row['id']}",
                        content=content,
                        source=DataSource.CHARACTER_CARDS,
                        metadata={
                            "name": name,
                            "creator": row.get("creator"),
                            "version": row.get("version"),
                            "type": "character_card",
                            "source": "characters",
                        },
                        score=float(score_val),
                    )
                    documents.append(doc)

                if include_chats:
                    limit_msgs = max(1, self.config.max_results // 2)
                    msg_rows = self.chacha_db.search_messages_by_content(query, limit=limit_msgs)
                    for row in msg_rows:
                        character_name = None
                        character_id = None
                        conv_id = row.get("conversation_id")
                        if conv_id:
                            conv = self.chacha_db.get_conversation_by_id(conv_id)
                            if conv and conv.get("character_id") is not None:
                                character_id_val = conv.get("character_id")
                                character_id_int = _coerce_int(character_id_val)
                                if character_id_int is not None:
                                    character_id = character_id_int
                                    card = self.chacha_db.get_character_card_by_id(character_id_int)
                                    if card:
                                        character_name = card.get("name")

                        content = f"{row.get('sender')}: {row.get('content', '')}"
                        metadata = {
                            "sender": row.get("sender"),
                            "timestamp": row.get("timestamp"),
                            "character_id": character_id,
                            "character_name": character_name,
                            "type": "chat_message",
                            "source": "characters",
                        }
                        doc = Document(
                            id=f"chat_{row['id']}",
                            content=content,
                            source=DataSource.CHARACTER_CARDS,
                            metadata=metadata,
                            score=0.5,
                        )
                        documents.append(doc)

            except (AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError) as e:
                if getattr(self.chacha_db, "backend_type", None) == BackendType.POSTGRESQL:
                    logger.warning(
                        'ChaCha backend search failed under PostgreSQL backend; skipping legacy fallback: {}',
                        e,
                    )
                    raise
                logger.debug(f"ChaCha backend search failed; falling back to legacy SQL: {e}")
            else:
                documents.sort(key=lambda x: x.score, reverse=True)
                logger.debug(f"Retrieved {len(documents)} Character/Chat documents via ChaCha backend")
                return documents

        # Legacy SQLite-style path
        # Search character cards
        card_sql = """
            SELECT
                cc.id,
                cc.name,
                cc.description,
                cc.personality,
                cc.first_message,
                cc.system_prompt,
                cc.scenario,
                cc.creator,
                cc.version
            FROM character_cards cc
            WHERE cc.deleted = 0 AND (
                cc.name LIKE ?
                OR cc.description LIKE ?
                OR cc.personality LIKE ?
                OR cc.scenario LIKE ?
                OR cc.system_prompt LIKE ?
            )
            LIMIT ?
        """
        params = [f"%{query}%"] * 5 + [self.config.max_results // 2]
        card_results = self._execute_query(card_sql, tuple(params))
        for row in card_results:
            content = f"""# {row['name']}\n\n**Description:** {row['description']}\n\n**Personality:** {row['personality']}\n\n**Scenario:** {row['scenario']}\n\n**First Message:** {row['first_message']}"""
            matches = sum([
                query.lower() in (row[field] or "").lower()
                for field in ["name", "description", "personality", "scenario", "system_prompt"]
            ])
            score = matches / 5.0 if matches else 0.0
            doc = Document(
                id=f"character_{row['id']}",
                content=content,
                source=DataSource.CHARACTER_CARDS,
                metadata={
                    "name": row["name"],
                    "creator": row["creator"],
                    "version": row["version"],
                    "type": "character_card",
                    "source": "characters",
                },
                score=score,
            )
            documents.append(doc)

        if include_chats:
            chat_sql = """
                SELECT
                    m.id,
                    m.content,
                    m.sender,
                    m.timestamp,
                    conv.character_id,
                    cc.name as character_name
                FROM messages m
                JOIN conversations conv ON m.conversation_id = conv.id
                LEFT JOIN character_cards cc ON conv.character_id = cc.id
                WHERE m.deleted = 0 AND m.content LIKE ?
                ORDER BY m.timestamp DESC
                LIMIT ?
            """
            chat_params = [f"%{query}%", self.config.max_results // 2]
            chat_results = self._execute_query(chat_sql, tuple(chat_params))
            for row in chat_results:
                content = f"[{row['sender']}]: {row['content']}"
                doc = Document(
                    id=f"chat_{row['id']}",
                    content=content,
                    source=DataSource.CHAT_HISTORY,
                    metadata={
                        "sender": row["sender"],
                        "timestamp": row["timestamp"],
                        "character": row["character_name"],
                        "type": "chat_message",
                        "source": "characters",
                    },
                    score=0.5,
                )
                documents.append(doc)

        documents.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"Retrieved {len(documents)} documents from Character Cards")
        return documents

    async def get_metadata(self, doc_id: str) -> dict[str, Any]:
        """Get metadata for a character card or chat."""
        if doc_id.startswith("character_"):
            card_id = doc_id.replace("character_", "")
            sql = "SELECT * FROM character_cards WHERE id = ?"
            results = self._execute_query(sql, (card_id,))
        elif doc_id.startswith("chat_"):
            chat_id = doc_id.replace("chat_", "")
            sql = """
                SELECT cm.*, cs.character_id, cc.name as character_name
                FROM chat_messages cm
                JOIN chat_sessions cs ON cm.session_id = cs.id
                LEFT JOIN character_cards cc ON cs.character_id = cc.id
                WHERE cm.id = ?
            """
            results = self._execute_query(sql, (chat_id,))
        else:
            return {}

        if results:
            return dict(results[0])

        return {}


class _PassThroughSqlGenerator:
    """Temporary SQL generator used by SQLRetriever until NL generation is added."""

    async def generate(self, *, query: str, target_id: str) -> dict[str, str]:
        _ = target_id
        sql = str(query or "").strip()
        if not sql:
            raise ValueError("Query must not be empty")
        if not sql.lower().startswith(("select", "with")):
            raise ValueError("sql_generation_failed: provide SQL beginning with SELECT or WITH")
        return {"sql": sql}


class SQLRetriever(BaseRetriever):
    """Retriever that executes read-only SQL and maps result rows to RAG documents."""

    def __init__(
        self,
        db_path: Optional[str],
        config: Optional[RetrievalConfig] = None,
        *,
        service: Optional[Any] = None,
        target_id: str = "media_db",
        timeout_ms: int = 5000,
        max_rows: int = 100,
        allow_nl_generation: bool = False,
    ) -> None:
        super().__init__(db_path, config, db_adapter=service)
        self.target_id = target_id
        self.timeout_ms = int(timeout_ms)
        self.max_rows = int(max_rows)
        self.allow_nl_generation = bool(allow_nl_generation)
        self._service = service or self._build_default_service()

    @staticmethod
    def _looks_like_sql(query: str) -> bool:
        return bool(_SQL_INPUT_PREFIX_RE.match(str(query or "")))

    def _emit_counter(self, outcome: str, reason: Optional[str] = None) -> None:
        labels: dict[str, str] = {"outcome": str(outcome)}
        if reason:
            labels["reason"] = str(reason)
        try:
            from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter

            increment_counter("rag_sql_retriever_requests_total", labels=labels)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass

    def _emit_duration(self, duration_seconds: float, outcome: str) -> None:
        try:
            from tldw_Server_API.app.core.Metrics.metrics_manager import observe_histogram

            observe_histogram(
                "rag_sql_retriever_duration_seconds",
                value=max(0.0, float(duration_seconds)),
                labels={"outcome": str(outcome)},
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass

    def _emit_rows(self, row_count: int, outcome: str) -> None:
        try:
            from tldw_Server_API.app.core.Metrics.metrics_manager import observe_histogram

            observe_histogram(
                "rag_sql_retriever_rows_returned",
                value=max(0.0, float(row_count)),
                labels={"outcome": str(outcome)},
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass

    def _build_default_service(self) -> Optional[Any]:
        if not self.db_path:
            return None
        try:
            from tldw_Server_API.app.core.Text2SQL.executor import SqliteReadOnlyExecutor
            from tldw_Server_API.app.core.Text2SQL.service import Text2SQLCoreService
            return Text2SQLCoreService(
                generator=_PassThroughSqlGenerator(),
                executor=SqliteReadOnlyExecutor(self.db_path),
            )
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None

    async def retrieve(self, query: str, **kwargs) -> list[Document]:
        started = time.perf_counter()
        query_text = str(query or "").strip()
        service = self._service
        if service is None:
            self._emit_counter("unavailable", reason="service_missing")
            self._emit_duration(time.perf_counter() - started, outcome="unavailable")
            return []
        if not query_text:
            self._emit_counter("empty_query")
            self._emit_duration(time.perf_counter() - started, outcome="empty_query")
            return []
        if not self.allow_nl_generation and not self._looks_like_sql(query_text):
            self._emit_counter("non_sql_input")
            self._emit_duration(time.perf_counter() - started, outcome="non_sql_input")
            return []

        target_id = str(kwargs.get("sql_target_id") or self.target_id)
        timeout_ms = kwargs.get("timeout_ms", self.timeout_ms)
        max_rows = kwargs.get("max_rows", min(self.max_rows, int(self.config.max_results or self.max_rows)))
        try:
            timeout_ms = int(timeout_ms)
        except (TypeError, ValueError):
            timeout_ms = self.timeout_ms
        try:
            max_rows = int(max_rows)
        except (TypeError, ValueError):
            max_rows = min(self.max_rows, int(self.config.max_results or self.max_rows))

        try:
            result = await service.generate_and_execute(
                query=query_text,
                target_id=target_id,
                timeout_ms=timeout_ms,
                max_rows=max_rows,
            )
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as exc:
            self._emit_counter("error", reason=type(exc).__name__)
            self._emit_duration(time.perf_counter() - started, outcome="error")
            logger.debug(f"SQL retrieval failed: {exc}")
            return []

        rows = result.get("rows", [])
        if not isinstance(rows, list):
            return []
        columns = [str(value) for value in result.get("columns", [])]

        docs: list[Document] = []
        capped_rows = rows[: int(self.config.max_results or len(rows))]
        total = max(1, len(capped_rows))
        for index, row in enumerate(capped_rows):
            if isinstance(row, dict):
                row_data = {str(key): value for key, value in row.items()}
            else:
                row_data = {"value": row}
            content = json.dumps(row_data, ensure_ascii=True, sort_keys=True)
            score = 1.0 - (float(index) / float(total))
            docs.append(
                Document(
                    id=f"sql:{target_id}:{index}",
                    content=content,
                    source=DataSource.SQL,
                    metadata={
                        "source": "sql",
                        "sql": str(result.get("sql", "")),
                        "columns": columns,
                        "target_id": target_id,
                        "row_index": index,
                        "row_count": int(result.get("row_count", len(capped_rows))),
                        "guardrail": result.get("guardrail", {}),
                        "truncated": bool(result.get("truncated", False)),
                    },
                    score=score,
                )
            )
        self._emit_counter("success")
        self._emit_duration(time.perf_counter() - started, outcome="success")
        self._emit_rows(len(docs), outcome="success")
        return docs

    async def get_metadata(self, doc_id: str) -> dict[str, Any]:
        return {"doc_id": doc_id, "source": "sql", "target_id": self.target_id}


class MultiDatabaseRetriever:
    """Orchestrates retrieval across multiple databases."""

    def __init__(
        self,
        db_paths: dict[str, str],
        user_id: str = "0",
        *,
        media_db: Optional[Any] = None,
        chacha_db: Optional[Any] = None,
        prompts_db: Optional[Any] = None,
        sql_retriever: Optional[BaseRetriever] = None,
    ):
        """
        Initialize multi-database retriever.

        Args:
            db_paths: Mapping of database names to paths
            user_id: User ID for vector store access
        """
        self.retrievers: dict[DataSource, BaseRetriever] = {}

        # Initialize retrievers for available databases
        media_db_path = db_paths.get("media_db")
        if media_db_path is not None or media_db is not None:
            self.retrievers[DataSource.MEDIA_DB] = MediaDBRetriever(
                media_db_path,
                user_id=user_id,
                media_db=media_db,
            )

        if "notes_db" in db_paths:
            self.retrievers[DataSource.NOTES] = NotesDBRetriever(
                db_paths["notes_db"],
                chacha_db=chacha_db,
            )

        if "prompts_db" in db_paths or prompts_db is not None:
            self.retrievers[DataSource.PROMPTS] = PromptsDBRetriever(
                db_paths.get("prompts_db"),
                prompts_db=prompts_db,
            )

        character_db_path = db_paths.get("character_cards_db") or db_paths.get("notes_db")
        if character_db_path:
            self.retrievers[DataSource.CHARACTER_CARDS] = CharacterCardsRetriever(
                character_db_path,
                chacha_db=chacha_db,
            )
            self.retrievers[DataSource.CHAT_HISTORY] = ChatHistoryRetriever(
                character_db_path,
                chacha_db=chacha_db,
            )
        world_books_db_path = db_paths.get("world_books_db") or character_db_path
        if world_books_db_path:
            self.retrievers[DataSource.WORLD_BOOKS] = WorldBooksRetriever(
                world_books_db_path,
                db_adapter=chacha_db,
            )
        dictionaries_db_path = db_paths.get("chat_dictionaries_db") or character_db_path
        if dictionaries_db_path:
            self.retrievers[DataSource.DICTIONARIES] = ChatDictionariesRetriever(
                dictionaries_db_path,
                db_adapter=chacha_db,
            )
        if "kanban_db" in db_paths:
            self.retrievers[DataSource.KANBAN] = KanbanDBRetriever(
                db_paths["kanban_db"],
                user_id=user_id,
            )
        # Optional: Claims retriever if provided
        if "claims_db" in db_paths:
            try:
                self.retrievers[DataSource.CLAIMS] = ClaimsRetriever(db_paths["claims_db"])
            except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
                logger.debug(f"ClaimsRetriever init skipped: {e}")
        if sql_retriever is not None:
            self.retrievers[DataSource.SQL] = sql_retriever

    # Resource management
    def close(self) -> None:
        try:
            for retr in list(self.retrievers.values()):
                close_fn = getattr(retr, "close", None)
                if callable(close_fn):
                    with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
                        close_fn()
        finally:
            self.retrievers.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
            self.close()
        # Do not suppress exceptions
        return False

    def __del__(self):
        with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
            self.close()

    async def retrieve(
        self,
        query: str,
        *,
        sources: Optional[list[DataSource]] = None,
        config: Optional[RetrievalConfig] = None,
        index_namespace: Optional[str] = None,
        retrieval_plan: Optional[RetrievalPlan] = None,
        # Optional per-source restrictions
        allowed_media_ids: Optional[list[int]] = None,
        allowed_note_ids: Optional[list[str]] = None,
    ) -> list[Document]:
        """
        Retrieve documents from one or more configured data sources.

        Args:
            query: The search query
            sources: Optional explicit list of `DataSource` to query. Defaults to all configured.
            config: Optional `RetrievalConfig` to apply to each retriever
            index_namespace: Optional namespace for vector stores

        Returns:
            A list of `Document` objects sorted by score (desc), capped by config.max_results if provided.
        """
        plan_sources: list[DataSource] = []
        if retrieval_plan is not None:
            plan_sources = _normalize_plan_sources(retrieval_plan)
            if plan_sources:
                sources = plan_sources
            config = _config_from_retrieval_plan(config, retrieval_plan)
            index_namespace = _index_namespace_from_retrieval_plan(
                retrieval_plan,
                plan_sources,
            )

        # Normalize the sources list
        ds_list: list[DataSource]
        if sources is None:
            ds_list = list(self.retrievers.keys())
        else:
            # Allow callers to pass strings; normalize to DataSource
            ds_list = []
            for s in sources:
                if isinstance(s, DataSource):
                    ds_list.append(s)
                else:
                    try:
                        ds_list.append(DataSource(str(s)))
                    except (ValueError, TypeError):
                        continue

        documents: list[Document] = []
        tasks: list[Any] = []

        async def _run_with_config(
            retriever: BaseRetriever,
            operation: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if config is None:
                return await operation(*args, **kwargs)
            operation_name = getattr(operation, "__name__", None)
            if isinstance(operation_name, str) and hasattr(retriever, operation_name):
                call_retriever = copy.copy(retriever)
                call_retriever.config = config
                return await getattr(call_retriever, operation_name)(*args, **kwargs)
            previous_config = getattr(retriever, "config", None)
            retriever.config = config
            try:
                return await operation(*args, **kwargs)
            finally:
                retriever.config = previous_config

        # Prepare async tasks for each source
        for src in ds_list:
            retr = self.retrievers.get(src)
            if retr is None:
                continue

            # Prefer hybrid/vector when requested and available for Media DB
            if (
                isinstance(retr, MediaDBRetriever)
                and config is not None
                and getattr(config, "use_vector", False)
                and getattr(config, "use_fts", False)
                and hasattr(retr, "retrieve_hybrid")
            ):
                tasks.append(_run_with_config(
                    retr,
                    retr.retrieve_hybrid,
                    query=query,
                    index_namespace=index_namespace,
                    allowed_media_ids=allowed_media_ids,
                ))
            elif (
                isinstance(retr, MediaDBRetriever)
                and config is not None
                and getattr(config, "use_vector", False)
                and hasattr(retr, "_retrieve_vector")
            ):
                tasks.append(_run_with_config(
                    retr,
                    retr._retrieve_vector,
                    query,
                    index_namespace=index_namespace,
                    allowed_media_ids=allowed_media_ids,
                ))
            elif (
                isinstance(retr, MediaDBRetriever)
                and config is not None
                and getattr(config, "use_fts", True)
                and hasattr(retr, "_retrieve_fts")
            ):
                tasks.append(_run_with_config(
                    retr,
                    retr._retrieve_fts,
                    query,
                    allowed_media_ids=allowed_media_ids,
                ))
            else:
                # Generic retrieve; pass through allowed IDs where applicable
                if isinstance(retr, NotesDBRetriever):
                    tasks.append(_run_with_config(
                        retr,
                        retr.retrieve,
                        query,
                        allowed_note_ids=allowed_note_ids,
                    ))
                elif isinstance(retr, MediaDBRetriever):
                    tasks.append(_run_with_config(
                        retr,
                        retr.retrieve,
                        query,
                        allowed_media_ids=allowed_media_ids,
                    ))
                elif isinstance(retr, CharacterCardsRetriever):
                    tasks.append(_run_with_config(
                        retr,
                        retr.retrieve,
                        query,
                        include_chats=False,
                    ))
                else:
                    tasks.append(_run_with_config(retr, retr.retrieve, query))

        # Execute all retrievals concurrently
        if tasks:
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            except (RuntimeError, TypeError, ValueError) as e:
                logger.error(f"Multi-database retrieval failed: {e}")
                results = []
        else:
            results = []

        # Flatten and filter out failures
        for res in results:
            if isinstance(res, Exception):
                # Skip failed sources (partial success expected)
                continue
            if isinstance(res, list):
                documents.extend(res)

        # Sort globally by score desc and cap by max_results
        documents.sort(key=lambda d: getattr(d, "score", 0.0), reverse=True)
        if config is not None and getattr(config, "max_results", None):
            documents = documents[: int(config.max_results)]

        return documents

    async def retrieve_from_plan(
        self,
        plan: RetrievalPlan,
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve documents using a normalized retrieval plan."""

        return await self.retrieve(
            plan.query,
            retrieval_plan=plan,
            **kwargs,
        )

    async def retrieve_with_fusion(
        self,
        query: str,
        *,
        sources: Optional[list[DataSource]] = None,
        fusion_method: str = "rrf",
    ) -> list[Document]:
        """Retrieve from multiple sources and fuse results."""
        # Collect per-source results
        source_results: dict[DataSource, list[Document]] = {}
        ds_list = list(self.retrievers.keys()) if sources is None else sources
        for src in ds_list:
            retr = self.retrievers.get(src)
            if retr is None:
                continue
            try:
                docs = await retr.retrieve(query)
            except (AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError):
                docs = []
            source_results[src] = docs

        # Apply fusion
        if fusion_method == "rrf":
            return self._reciprocal_rank_fusion(source_results)
        if fusion_method == "weighted":
            return self._weighted_fusion(source_results)
        if fusion_method == "max":
            return self._max_fusion(source_results)

        # Default: simple concatenation
        all_docs: list[Document] = []
        for docs in source_results.values():
            all_docs.extend(docs)
        return sorted(all_docs, key=lambda x: getattr(x, "score", 0.0), reverse=True)

    def _reciprocal_rank_fusion(
        self,
        source_results: dict[DataSource, list[Document]],
        k: int = 60,
    ) -> list[Document]:
        doc_scores: dict[str, dict[str, Any]] = {}
        for _source, docs in source_results.items():
            for rank, doc in enumerate(docs, 1):
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = {"doc": doc, "score": 0.0}
                doc_scores[doc.id]["score"] += 1.0 / (k + rank)

        fused_docs: list[Document] = [
            Document(
                id=item["doc"].id,
                content=item["doc"].content,
                source=item["doc"].source,
                metadata=item["doc"].metadata,
                score=float(item["score"]),
            )
            for item in sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        ]
        return fused_docs

    def _weighted_fusion(
        self,
        source_results: dict[DataSource, list[Document]],
        weights: Optional[dict[DataSource, float]] = None,
    ) -> list[Document]:
        weights = weights or {
            DataSource.MEDIA_DB: 1.0,
            DataSource.NOTES: 0.8,
            DataSource.PROMPTS: 0.6,
            DataSource.CHARACTER_CARDS: 0.5,
            DataSource.KANBAN: 0.85,
            DataSource.SQL: 0.9,
        }
        doc_scores: dict[str, dict[str, Any]] = {}
        for source, docs in source_results.items():
            w = weights.get(source, 1.0)
            for doc in docs:
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = {"doc": doc, "score": 0.0}
                doc_scores[doc.id]["score"] += float(getattr(doc, "score", 0.0)) * w
        fused_docs: list[Document] = [
            Document(
                id=item["doc"].id,
                content=item["doc"].content,
                source=item["doc"].source,
                metadata=item["doc"].metadata,
                score=float(item["score"]),
            )
            for item in sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        ]
        return fused_docs

    def _max_fusion(
        self,
        source_results: dict[DataSource, list[Document]],
    ) -> list[Document]:
        doc_map: dict[str, Document] = {}
        for _source, docs in source_results.items():
            for doc in docs:
                existing = doc_map.get(doc.id)
                if existing is None or float(getattr(doc, "score", 0.0)) > float(
                    getattr(existing, "score", 0.0)
                ):
                    doc_map[doc.id] = doc
        return sorted(doc_map.values(), key=lambda d: getattr(d, "score", 0.0), reverse=True)

class ClaimsRetriever(BaseRetriever):
    """Retriever for Claims table (ingestion-time factual statements)."""

    def __init__(
        self,
        db_path: Optional[str],
        config: Optional[RetrievalConfig] = None,
        *,
        media_db: Optional[Any] = None
    ) -> None:
        super().__init__(db_path, config, db_adapter=media_db)
        attached = None
        own = False
        if media_db is None:
            attached = self._maybe_attach_media_db(self.db_path)
            own = attached is not None
        self.media_db = media_db or attached
        self._db_adapter = self.media_db
        self._own_media_db = own

    def _maybe_attach_media_db(self, db_path: Optional[str]):
        if not db_path:
            return None
        try:
            # Defensive re-validation in case callers bypass BaseRetriever.
            validated_path = self._validate_path(db_path)
            if not validated_path:
                return None
            return create_media_database("rag_service", db_path=validated_path)
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            sqlite3.Error,
        ):
            return None

    def close(self):
        try:
            if self._own_media_db and self.media_db is not None:
                close_fn = getattr(self.media_db, 'close_connection', None)
                if callable(close_fn):
                    close_fn()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    async def retrieve(self, query: str, **kwargs) -> list[Document]:
        if self.media_db is not None:
            return self._retrieve_via_media_backend(query)

        documents: list[Document] = []
        try:
            # Try FTS on claims_fts first
            sql = (
                "SELECT c.id, c.media_id, c.chunk_index, c.claim_text "
                "FROM claims_fts JOIN Claims c ON claims_fts.rowid = c.id "
                "WHERE claims_fts MATCH ? AND c.deleted = 0 LIMIT ?"
            )
            params = (query, int(self.config.max_results))
            rows = self._execute_query(sql, params)
            for r in rows:
                doc_id = f"claim_{r['id']}"
                content = r["claim_text"]
                documents.append(
                    Document(
                        id=doc_id,
                        content=content,
                        metadata={
                            "media_id": r["media_id"],
                            "chunk_index": r["chunk_index"],
                            "source": "claim",
                        },
                        source=DataSource.CLAIMS,
                        score=0.6,
                    )
                )
            if not documents:
                # Fallback to LIKE if FTS returns no rows
                sql = (
                    "SELECT id, media_id, chunk_index, claim_text FROM Claims "
                    "WHERE deleted = 0 AND claim_text LIKE ? LIMIT ?"
                )
                params = (f"%{query}%", int(self.config.max_results))
                rows = self._execute_query(sql, params)
                for r in rows:
                    doc_id = f"claim_{r['id']}"
                    content = r["claim_text"]
                    documents.append(
                        Document(
                            id=doc_id,
                            content=content,
                            metadata={
                                "media_id": r["media_id"],
                                "chunk_index": r["chunk_index"],
                                "source": "claim",
                            },
                            source=DataSource.CLAIMS,
                            score=0.4,
                        )
                    )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
            logger.debug(f"Claims FTS failed, fallback to LIKE: {e}")
            sql = (
                "SELECT id, media_id, chunk_index, claim_text FROM Claims "
                "WHERE deleted = 0 AND claim_text LIKE ? LIMIT ?"
            )
            params = (f"%{query}%", int(self.config.max_results))
            rows = self._execute_query(sql, params)
            for r in rows:
                doc_id = f"claim_{r['id']}"
                content = r["claim_text"]
                documents.append(
                    Document(
                        id=doc_id,
                        content=content,
                        metadata={
                            "media_id": r["media_id"],
                            "chunk_index": r["chunk_index"],
                            "source": "claim",
                        },
                        source=DataSource.CLAIMS,
                        score=0.4,
                    )
                )
        return documents

    def _retrieve_via_media_backend(self, query: str) -> list[Document]:
        if self.media_db is None:
            return []
        try:
            results = self.media_db.search_claims(query, limit=int(self.config.max_results))
        except (AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.error(f"MediaDatabase claims search failed: {exc}")
            return []
        documents: list[Document] = []
        min_score = float(self.config.min_score or 0.0)
        backend_type = getattr(self.media_db, 'backend_type', None)
        for row in results:
            score = row.get('relevance_score') if isinstance(row, dict) else None
            try:
                score_val = float(score) if score is not None else 0.0
            except (TypeError, ValueError):
                score_val = 0.0
            if backend_type == BackendType.SQLITE:
                # SQLite bm25 returns lower (often negative) values for better matches.
                score_val = -score_val
            if score_val < min_score:
                continue
            metadata = {
                "media_id": row.get("media_id"),
                "chunk_index": row.get("chunk_index"),
                "source": "claim",
            }
            documents.append(
                Document(
                    id=f"claim_{row.get('id')}",
                    content=row.get('claim_text') or '',
                    metadata=metadata,
                    source=DataSource.CLAIMS,
                    score=score_val if score_val else 0.4,
                )
            )
        return documents

    async def get_metadata(self, doc_id: str) -> dict[str, Any]:
        try:
            cid = doc_id.replace("claim_", "")
            sql = "SELECT * FROM Claims WHERE id = ?"
            rows = self._execute_query(sql, (cid,))
            return dict(rows[0]) if rows else {}
        except (AttributeError, TypeError, ValueError):
            return {}

    # (no second retrieve method inside ClaimsRetriever)

# ---------------------------------------------------------------------------
# Backward compatibility aliases for test suites expecting older names
try:
    # Some tests import MediaDatabaseRetriever; map to MediaDBRetriever for compatibility
    MediaDatabaseRetriever = MediaDBRetriever
except (NameError, AttributeError):
    pass


# Pipeline integration function
async def retrieve_from_databases(context: Any, **kwargs) -> Any:
    """Retrieve documents from configured databases for pipeline."""
    config = context.config.get("database_retrieval", {})

    # Get database paths from config
    db_paths = config.get("db_paths", {})
    if not db_paths:
        logger.warning("No database paths configured")
        return context

    # Create multi-database retriever
    retriever = MultiDatabaseRetriever(db_paths)

    # Configure retrieval
    retrieval_config = RetrievalConfig(
        max_results=config.get("max_results", 20),
        min_score=config.get("min_score", 0.0),
        use_fts=config.get("use_fts", True),
        include_metadata=config.get("include_metadata", True)
    )

    # Get sources to search
    sources = config.get("sources")
    if sources:
        sources = [DataSource[s.upper()] for s in sources]

    # Retrieve with fusion if enabled
    if config.get("use_fusion", True):
        documents = await retriever.retrieve_with_fusion(
            query=context.query,
            sources=sources,
            fusion_method=config.get("fusion_method", "rrf")
        )
    else:
        documents = await retriever.retrieve(
            query=context.query,
            sources=sources,
            config=retrieval_config
        )

    # Update context
    context.documents = documents
    context.metadata["database_retrieval"] = {
        "sources_searched": [s.value for s in (sources or retriever.retrievers.keys())],
        "documents_retrieved": len(documents),
        "fusion_used": config.get("use_fusion", True)
    }

    logger.info(f"Retrieved {len(documents)} documents from databases")

    return context
