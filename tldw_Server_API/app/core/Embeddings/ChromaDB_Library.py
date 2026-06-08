# ChromaDB_Library.py
# Description: Functions for managing embeddings in ChromaDB
#
# Imports:
import os
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

# Redundant in some environments, but keep explicit Dict import for clarity
from typing import Any, Callable, Literal, Optional, Union

# 3rd-Party Imports:
import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.api.types import QueryResult
from chromadb.errors import ChromaError

_CHROMA_IMPORT_EXCEPTIONS = (ImportError, AttributeError, RuntimeError)
_CHROMA_EMBEDDINGS_IMPORT_EXCEPTIONS = (ImportError, OSError, RuntimeError)

try:
    # Prefer the modern Settings from chromadb.config (works in 0.4.x and 1.x)
    from chromadb.config import Settings as ChromaSettings  # type: ignore
except _CHROMA_IMPORT_EXCEPTIONS:  # pragma: no cover - fallback for older versions
    from chromadb import Settings as ChromaSettings  # type: ignore

_CHROMA_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    ChromaError,
)


def _is_transient_chroma_segment_error(error: BaseException) -> bool:
    """Return True for Chroma's transient HNSW segment-reader race."""
    message = str(error).lower()
    return "hnsw segment reader" in message and "nothing found on disk" in message

#
# Local Imports:
from tldw_Server_API.app.core.Chunking import Chunker, chunk_for_embedding  # Using V2 through compatibility layer
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database

# Import embeddings creation lazily/safely to avoid hard dependency at import time
try:
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
        create_embedding,
        create_embeddings_batch,
    )
    _EMBEDDINGS_BACKEND_AVAILABLE = True
except _CHROMA_EMBEDDINGS_IMPORT_EXCEPTIONS:
    _EMBEDDINGS_BACKEND_AVAILABLE = False
    def create_embedding(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("Embeddings backend unavailable; install embeddings dependencies")
    def create_embeddings_batch(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("Embeddings backend unavailable; install embeddings dependencies")
import contextlib

from tldw_Server_API.app.core.Embeddings.audit_adapter import (
    log_security_violation,
)
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze  # Assuming this is correct
from tldw_Server_API.app.core.testing import env_flag_enabled
from tldw_Server_API.app.core.Utils.prompt_loader import load_prompt
from tldw_Server_API.app.core.Utils.Utils import logger  # Assuming this is 'logging' aliased or a custom logger


#
#######################################################################################################################
#
# Security Functions:
def validate_user_id(user_id: str) -> str:
    """
    Validates and sanitizes user_id to prevent path traversal attacks.

    Args:
        user_id: The user identifier to validate

    Returns:
        Sanitized user_id

    Raises:
        ValueError: If user_id contains invalid characters or patterns
    """
    if not user_id:
        raise ValueError("user_id cannot be empty")

    # Convert to string
    raw_user_id = str(user_id)
    # First, check raw input for forbidden characters (before trimming)
    if any(pattern in raw_user_id for pattern in ['..', '/', '\\', '\x00', '\n', '\r']):
        logger.error(f"Potential path traversal attempt detected in user_id: {user_id[:50]}")
        # Best-effort unified audit (non-blocking)
        log_security_violation(user_id=raw_user_id[:50], action="path_traversal_attempt", metadata={"attempted_value": raw_user_id[:100]})
        raise ValueError("Invalid user_id: contains forbidden characters")

    # Now trim safe leading/trailing whitespace
    user_id = raw_user_id.strip()

    # Only allow alphanumeric, underscore, and hyphen
    if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
        logger.error(f"Invalid user_id format: {user_id[:50]}")
        log_security_violation(user_id=user_id[:50], action="invalid_user_id", metadata={"reason": "invalid_characters"})
        raise ValueError("Invalid user_id: must contain only alphanumeric characters, underscores, and hyphens")

    # Limit length to prevent DoS
    if len(user_id) > 255:
        raise ValueError("Invalid user_id: exceeds maximum length of 255 characters")

    return user_id

def validate_model_id(model_id: str) -> str:
    """
    Validates model identifier to prevent injection attacks.

    Note: This function allows forward slashes to support model paths like
    "org/model" (e.g., "openai/text-embedding-3-small"). It should NOT be used
    to validate identifiers that will be used in file path construction.
    For path-safe identifiers, use validate_user_id() instead.

    Args:
        model_id: The model identifier to validate

    Returns:
        Validated model_id

    Raises:
        ValueError: If model_id contains invalid patterns
    """
    if not model_id:
        raise ValueError("model_id cannot be empty")

    model_id = str(model_id).strip()

    # Allow forward slash for model paths like "org/model" but prevent traversal
    if '..' in model_id or model_id.startswith('/') or '\\' in model_id:
        logger.error(f"Invalid model_id format: {model_id[:100]}")
        raise ValueError("Invalid model_id: contains forbidden patterns")

    # Allow alphanumeric, underscore, hyphen, forward slash, and dot
    if not re.match(r'^[a-zA-Z0-9_\-/\.]+$', model_id):
        raise ValueError("Invalid model_id: contains invalid characters")

    if len(model_id) > 500:
        raise ValueError("Invalid model_id: exceeds maximum length")

    return model_id


def _coerce_config_dict(raw: Any) -> dict[str, Any]:
    """Best-effort conversion of config-like objects into plain dicts."""
    if isinstance(raw, dict):
        return raw.copy()
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(raw, attr, None)
        if callable(fn):
            try:
                return dict(fn())
            except _CHROMA_NONCRITICAL_EXCEPTIONS:
                pass
    items = getattr(raw, "items", None)
    if callable(items):
        try:
            return dict(items())
        except _CHROMA_NONCRITICAL_EXCEPTIONS:
            return {}
    return {}


def _normalize_user_embedding_config(raw_config: Any) -> dict[str, Any]:
    """Ensure user_embedding_config always includes an embedding_config mapping."""
    cfg = _coerce_config_dict(raw_config)
    embedding_config = cfg.get("embedding_config")
    if embedding_config is None:
        embedding_config = cfg.get("EMBEDDING_CONFIG")
    if embedding_config is None:
        reserved = {
            "USER_DB_BASE_DIR",
            "chroma_client_settings",
            "openai_api",
            "local_api",
            "cohere_api",
            "google_api",
            "voyage_api",
            "mistral_api",
        }
        embedding_config = {k: v for k, v in cfg.items() if k not in reserved}
    if embedding_config is None:
        embedding_config = {}
    if not isinstance(embedding_config, dict):
        embedding_config = _coerce_config_dict(embedding_config)
    normalized = cfg.copy()
    normalized["embedding_config"] = embedding_config
    return normalized


def _persistent_client_cache_key(user_chroma_path: Path) -> str:
    """Return a stable key for process-local PersistentClient reuse.

    `user_chroma_path` already comes from `DatabasePaths.get_user_chroma_dir()`,
    which resolves and constrains the path under the sanitized user storage root.
    Avoid re-resolving here so the cache key stays a pure normalization step.
    """
    return os.path.normcase(os.fspath(user_chroma_path))


@dataclass
class _PersistentClientState:
    """Track a shared PersistentClient plus its synchronization and lease count."""

    client: Any
    lock: threading.RLock
    ref_count: int = 1

#
# Functions:
ChromaIncludeLiteral = Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]

class ChromaDBManager:
    """
    Manages ChromaDB instances and operations for specific users.
    Each instance of this class corresponds to a user's isolated ChromaDB storage.
    """
    DEFAULT_COLLECTION_NAME_PREFIX = "user_embeddings_for_"  # Can be made configurable

    def __init__(
        self,
        user_id: str,
        user_embedding_config: dict[str, Any],
        *,
        client: Optional[Any] = None,
        client_factory: Optional[Callable[[Path, dict[str, Any]], Any]] = None,
    ):
        """
        Initializes the ChromaDBManager for a specific user.

        Args:
            user_id (str): The ID of the user for whom this ChromaDB instance is.
            user_embedding_config (Dict[str, Any]): The global application configuration dictionary.
        """
        if not user_embedding_config:
            logger.error("Initialization failed: user_embedding_config cannot be empty for ChromaDBManager.")
            raise ValueError("user_embedding_config cannot be empty for ChromaDBManager.")

        # Validate and sanitize user_id to prevent path traversal
        try:
            self.user_id = validate_user_id(user_id)
        except ValueError as e:
            logger.error(f"Initialization failed: {e}")
            raise

        self.user_embedding_config = _normalize_user_embedding_config(user_embedding_config)
        if not self.user_embedding_config:
            logger.error("Initialization failed: normalized user_embedding_config is empty.")
            raise ValueError("user_embedding_config cannot be empty for ChromaDBManager.")
        self._lock = threading.RLock()  # Rebound to a shared lock for cached persistent clients.
        self._persistent_client_cache_key: Optional[str] = None

        # --- Configuration Usage (Point 1) ---
        user_db_base_dir_str = self.user_embedding_config.get("USER_DB_BASE_DIR")
        if not user_db_base_dir_str:
            logger.critical("USER_DB_BASE_DIR not found in user_embedding_config. ChromaDBManager cannot be initialized.")
            raise ValueError("USER_DB_BASE_DIR not configured in application settings.")

        try:
            self.user_chroma_path = DatabasePaths.get_user_chroma_dir(
                self.user_id,
                base_dir_override=user_db_base_dir_str,
            )
        except _CHROMA_NONCRITICAL_EXCEPTIONS as e:
            logger.critical(
                f"Failed to resolve ChromaDB storage path for user '{self.user_id}': {e}",
                exc_info=True,
            )
            raise

        logger.info(f"ChromaDBManager for user '{self.user_id}' initialized. Path: {self.user_chroma_path}")

        chroma_client_settings_config = self.user_embedding_config.get("chroma_client_settings", {})

        # Initialize embedding configuration early so it's available regardless of backend choice
        self.embedding_config = self.user_embedding_config.get("embedding_config", {})
        self.default_embedding_model_id = self.embedding_config.get('default_model_id')
        # Backend selection and initialization
        if client is not None:
            # Constructor-injected client takes precedence
            self.client = client
            logger.info(f"User '{self.user_id}': Using injected Chroma client instance.")
        elif client_factory is not None:
            try:
                factory_settings: dict[str, Any] = {
                    **chroma_client_settings_config,
                    "persist_directory": str(self.user_chroma_path),
                }
                self.client = client_factory(self.user_chroma_path, factory_settings)
                logger.info(f"User '{self.user_id}': Created Chroma client via injected factory.")
            except _CHROMA_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"User '{self.user_id}': client_factory failed: {e}", exc_info=True)
                raise RuntimeError(f"Chroma client factory failed: {e}") from e
        else:
            backend = str(chroma_client_settings_config.get("backend", "persistent")).lower()
            # Honor explicit config, and also support CHROMADB_FORCE_STUB for tests/CI
            _env_force_stub = env_flag_enabled("CHROMADB_FORCE_STUB")
            use_stub = bool(
                chroma_client_settings_config.get("use_in_memory_stub", False)
                or backend == "stub"
                or _env_force_stub
            )
            allow_stub_fallback = bool(chroma_client_settings_config.get("allow_stub_fallback", True))

            if use_stub:
                # Scope the stub client key by user and base dir to avoid cross-config leakage
                stub_key = f"{self.user_id}::{str(user_db_base_dir_str)}"
                with _TEST_STUB_CLIENTS_LOCK:
                    cli = _TEST_STUB_CLIENTS.get(stub_key)
                    if cli is None:
                        cli = _InMemoryChromaClient()
                        _TEST_STUB_CLIENTS[stub_key] = cli
                self.client = cli
                logger.warning(
                    f"User '{self.user_id}': Using internal in-memory Chroma client (config backend=stub)."
                )
            else:
                # Build robust Settings with explicit persist_directory for Chroma 0.4.x/1.x compatibility
                client_settings = ChromaSettings(
                    persist_directory=str(self.user_chroma_path),
                    anonymized_telemetry=chroma_client_settings_config.get("anonymized_telemetry", False),
                    allow_reset=chroma_client_settings_config.get("allow_reset", True),
                )
                cache_key = _persistent_client_cache_key(self.user_chroma_path)
                with _PERSISTENT_CLIENTS_LOCK:
                    cached_state = _PERSISTENT_CLIENTS.get(cache_key)
                    if cached_state is not None:
                        cached_state.ref_count += 1
                        self.client = cached_state.client
                        self._lock = cached_state.lock
                        self._persistent_client_cache_key = cache_key
                        logger.info(
                            f"User '{self.user_id}': Reusing cached Chroma persistent client for {self.user_chroma_path}."
                        )
                    else:
                        try:
                            self.client = chromadb.PersistentClient(
                                path=str(self.user_chroma_path),
                                settings=client_settings,
                            )
                        except _CHROMA_NONCRITICAL_EXCEPTIONS as e:
                            logger.error(
                                f"Failed to initialize Chroma persistent client at {self.user_chroma_path}: {e}",
                                exc_info=True,
                            )
                            if allow_stub_fallback:
                                stub_key = f"{self.user_id}::{str(user_db_base_dir_str)}"
                                with _TEST_STUB_CLIENTS_LOCK:
                                    cli = _TEST_STUB_CLIENTS.get(stub_key)
                                    if cli is None:
                                        cli = _InMemoryChromaClient()
                                        _TEST_STUB_CLIENTS[stub_key] = cli
                                self.client = cli
                                logger.warning(
                                    f"User '{self.user_id}': Falling back to in-memory Chroma stub (allow_stub_fallback=true)."
                                )
                            else:
                                raise RuntimeError(f"ChromaDB client initialization failed: {e}") from e
                        else:
                            _PERSISTENT_CLIENTS[cache_key] = _PersistentClientState(
                                client=self.client,
                                lock=threading.RLock(),
                            )
                            self._lock = _PERSISTENT_CLIENTS[cache_key].lock
                            self._persistent_client_cache_key = cache_key

        # Default embedding model_id for this manager instance.
        # Already initialized above; keep values consistent for non-stub client path.

        if not self.default_embedding_model_id:
            logger.warning(  # Changed to warning, operations might still succeed if model_id is always overridden
                f"User '{self.user_id}': No 'default_model_id' found in 'embedding_config'. "
                "Operations will require explicit 'embedding_model_id_override'."
            )
            # Not raising an error to allow flexibility if all calls provide an override.

        model_details = self.embedding_config.get("models", {}).get(self.default_embedding_model_id, {})
        logger.info(
            f"User '{self.user_id}' ChromaDBManager configured. "
            f"Default Embedding Model ID: {self.default_embedding_model_id or 'Not Set (Override Required)'} "
            f"(Provider: {model_details.get('provider', 'N/A')}, Name: {model_details.get('model_name_or_path', 'N/A')})"
        )

    # Resource management helpers
    def close(self) -> None:
        """Close underlying ChromaDB client and release file descriptors."""
        with self._lock:
            client = getattr(self, "client", None)
            if client is None:
                return
            cache_key = getattr(self, "_persistent_client_cache_key", None)
            if cache_key:
                should_close_cached_client = False
                with _PERSISTENT_CLIENTS_LOCK:
                    cached_state = _PERSISTENT_CLIENTS.get(cache_key)
                    if cached_state is not None and cached_state.client is client:
                        cached_state.ref_count = max(0, cached_state.ref_count - 1)
                        if cached_state.ref_count == 0:
                            _PERSISTENT_CLIENTS.pop(cache_key, None)
                            should_close_cached_client = True
                    else:
                        should_close_cached_client = True

                if not should_close_cached_client:
                    self.client = None
                    self._persistent_client_cache_key = None
                    return
            try:
                # Prefer explicit close if available (future APIs)
                close_fn = getattr(client, "close", None)
                if callable(close_fn):
                    close_fn()
                else:
                    # ChromaDB PersistentClient uses an internal system service
                    system = getattr(client, "_system", None)
                    stop_fn = getattr(system, "stop", None) if system is not None else None
                    if callable(stop_fn):
                        stop_fn()
            except _CHROMA_NONCRITICAL_EXCEPTIONS as e:
                # Best-effort close; log and continue
                logger.warning(f"User '{self.user_id}': Error while closing ChromaDB client: {e}")
            finally:
                with contextlib.suppress(_CHROMA_NONCRITICAL_EXCEPTIONS):
                    self.client = None
                    self._persistent_client_cache_key = None

    # Support context manager usage
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        # Last-resort cleanup if user forgot to close
        with contextlib.suppress(_CHROMA_NONCRITICAL_EXCEPTIONS):
            self.close()

    def _batched(self, iterable, n):
        """Helper to yield batches from an iterable."""
        it = iter(iterable)
        while True:
            batch = list(islice(it, n))
            if not batch:
                return
            yield batch

    def _clean_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Cleans metadata to ensure compatibility with ChromaDB."""
        cleaned = {}
        if not isinstance(metadata, dict):
            logger.warning(
                f"User '{self.user_id}': Received non-dict metadata: {type(metadata)}. Returning empty dict.")
            return cleaned

        for key, value in metadata.items():
            if value is None:  # ChromaDB can handle None, but explicit skip or convert might be safer.
                # cleaned[key] = None # Or skip
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, (np.int32, np.int64, np.int16, np.int8)):
                cleaned[key] = int(value)
            elif isinstance(value, (np.float32, np.float64, np.float16)):
                cleaned[key] = float(value)
            elif isinstance(value, np.bool_):
                cleaned[key] = bool(value)
            elif isinstance(value, (list, tuple)):  # Chroma allows lists of primitives
                cleaned_list = [self._clean_metadata_value(v) for v in value]
                # Some chromadb builds crash on zero-length lists; drop them eagerly
                cleaned_list = [
                    item for item in cleaned_list
                    if item is not None and not (isinstance(item, str) and item == "")
                ]
                if cleaned_list:
                    cleaned[key] = cleaned_list
                else:
                    logger.debug(
                        f"User '{self.user_id}': Dropping metadata key '{key}' due to empty list value.")
            else:  # Fallback to string, log a warning for unexpected types
                logger.debug(
                    f"User '{self.user_id}': Converting metadata value of type {type(value)} for key '{key}' to string.")
                cleaned[key] = str(value)
        return cleaned

    def _clean_metadata_value(self, value: Any) -> Any:
        """Helper for cleaning individual values within a list in metadata."""
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (np.int32, np.int64, np.int16, np.int8)):
            return int(value)
        if isinstance(value, (np.float32, np.float64, np.float16)):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        logger.debug(f"User '{self.user_id}': Converting list element of type {type(value)} to string in metadata.")
        return str(value)

    def get_user_default_collection_name(self) -> str:
        """Gets the default collection name for the user, incorporating user ID."""
        # Sanitize user_id for collection name if it can contain special characters
        # For now, assuming user_id is safe or ChromaDB handles it.
        return f"{self.DEFAULT_COLLECTION_NAME_PREFIX}{self.user_id}"

    # Point 2: Collection Management# FIXME - Implement this
    # When creating a new collection for a specific model
    model_id_for_new_collection = "user_chosen_model_for_this_collection"
    # You'd need a way to get the dimension for this model_id, perhaps from Embeddings_Create or config
    # model_dimension = get_dimension_for_model(app_config, model_id_for_new_collection)
    # chroma_manager.get_or_create_collection(
    #     collection_name="my_new_collection_name",
    #     collection_metadata={
    #         "source_embedding_model_id": model_id_for_new_collection,
    #         "embedding_dimension": model_dimension,
    #         "hnsw:space": "cosine" # or other relevant Chroma params
    #     }
    # )
    # Then, when process_and_store_content or vector_search operate on a named collection, they should ideally:
    #
    #     Retrieve the collection.
    #
    #     Check its metadata for the source_embedding_model_id.
    #
    #     Use that model ID for embedding generation/querying, overriding the ChromaDBManager's default_embedding_model_id.
    #     This makes each collection self-contained regarding its embedding model.
    #
    # The current code in store_in_chroma has improved dimension checking and will recreate the collection if a dimension mismatch occurs, logging the embedding_model_id_for_dim_check. It also attempts to store the dimension in the collection metadata upon recreation.

    def get_or_create_collection(self, collection_name: Optional[str] = None,
                                  collection_metadata: Optional[dict[str, Any]] = None) -> Collection:
        """
        Gets or creates a ChromaDB collection.

        Args:
            collection_name (Optional[str]): Name of the collection. Defaults to user's default.
            collection_metadata (Optional[Dict[str, Any]]): Metadata for the collection,
                                                           e.g., {'hnsw:space': 'cosine'}.

        Returns:
            Collection: The ChromaDB collection object.

        Raises:
            RuntimeError: If collection creation or retrieval fails.
        """
        name_to_use = collection_name or self.get_user_default_collection_name()
        with self._lock:
            try:
                # The embedding_function parameter is for Chroma to generate embeddings.
                # Since we provide embeddings directly, it's not strictly needed here.
                # However, setting metadata like hnsw:space can be useful.
                cleaned = self._clean_metadata(collection_metadata) if collection_metadata else None
                try:
                    collection = self.client.get_or_create_collection(
                        name=name_to_use,
                        metadata=cleaned
                    )
                except TypeError as te:
                    # Older/simpler clients (including our test stub prior to this patch)
                    # may not accept a 'metadata' kwarg. Fallback to calling without it,
                    # then apply metadata via modify() if available.
                    if 'metadata' in str(te):
                        collection = self.client.get_or_create_collection(name=name_to_use)
                        if cleaned and hasattr(collection, 'modify'):
                            with contextlib.suppress(_CHROMA_NONCRITICAL_EXCEPTIONS):
                                collection.modify(metadata=cleaned)
                    else:
                        raise
                logger.info(f"User '{self.user_id}': Accessed/Created collection '{name_to_use}'.")
                return collection
            except _CHROMA_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"User '{self.user_id}': Failed to get or create collection '{name_to_use}': {e}",
                             exc_info=True)
                raise RuntimeError(f"Failed to access or create collection '{name_to_use}': {e}") from e

    # Point 4: Situate Context - generates a succinct header for each chunk using an LLM
    def situate_context(self,
                        provider: Optional[str] = None,
                        model_override: Optional[str] = None,
                        doc_content: str = "",
                        chunk_content: str = "",
                        temperature: Optional[float] = None,
                        **kwargs) -> str:
        """Generate a succinct context header situating a chunk within its document.

        Constructs a single custom prompt that includes both the document slice (or full doc/outline)
        and the chunk content, followed by the situate instruction. Uses the configured provider and
        optional model override via the unified analyze() function.
        """
        # Backward/alias compatibility: tests may pass api_name_for_context as the model name
        try:
            alias_model = kwargs.get("api_name_for_context")
            if alias_model and not model_override:
                model_override = alias_model
        except _CHROMA_NONCRITICAL_EXCEPTIONS:
            pass

        provider = provider or "openai"
        # Load instruction from prompts with safe fallback
        situate_instr = load_prompt("embeddings", "Situate Context Prompt") or (
            "Please give a short succinct context to situate this chunk within the overall document\n"
            "for the purposes of improving search retrieval of the chunk.\n"
            "Answer only with the succinct context and nothing else."
        )
        # Build a single user prompt that carries both doc and chunk
        custom_prompt = (
            f"<document>\n{doc_content}\n</document>\n\n"
            f"<chunk>\n{chunk_content}\n</chunk>\n\n"
            f"{situate_instr}"
        )
        try:
            resp = analyze(
                provider or "openai",
                "",  # keep input minimal; prompt includes all needed content
                custom_prompt,
                None,   # api_key (None -> load from config)
                "You generate concise context headers for retrieval.",
                (float(temperature) if temperature is not None else 0.1),
                streaming=False,
                recursive_summarization=False,
                chunked_summarization=False,
                chunk_options=None,
                model_override=model_override,
            )
            # analyze may return a generator; ensure string
            text = resp if isinstance(resp, str) else str(resp)
            return (text or "").strip()
        except Exception as e:
            logger.error(f"User '{self.user_id}': Error in situate_context with LLM '{provider}': {e}", exc_info=True)
            return ""

    def _estimate_token_count(self, text: str) -> int:
        """Rough token estimate without external tokenizer.

        Uses a simple heuristic: tokens ~= max(words, chars/4).
        """
        if not text:
            return 0
        words = text.split()
        return max(len(words), len(text) // 4)

    def _build_document_outline(self, provider: str, model_override: Optional[str], doc_content: str, temperature: Optional[float]) -> str:
        """Create a concise, high-level outline of the document using an LLM.

        Returns an outline as bullet points or short titled sections. On failure, returns empty string.
        """
        try:
            prompt = load_prompt("embeddings", "Document Outline Prompt") or (
                "Produce a brief outline of the document with 5-10 bullets. "
                "Each bullet should have a short section title and a one-line summary."
            )
            custom_prompt = f"<document>\n{doc_content}\n</document>\n\n{prompt}"
            resp = analyze(
                provider or "openai",
                "",
                custom_prompt,
                None,
                "You write concise document outlines.",
                (float(temperature) if temperature is not None else 0.1),
                False,
                False,
                False,
                None,
                model_override=model_override,
            )
            return (resp or "").strip()
        except _CHROMA_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"User '{self.user_id}': Outline generation failed: {e}")
            return ""

    def process_and_store_content(self,
                                  content: str,
                                  media_id: Union[int, str],  # TODO: Update type based on new MediaDatabase
                                  file_name: str,  # TODO: Get from new MediaDatabase if it stores this
                                  collection_name: Optional[str] = None,
                                  embedding_model_id_override: Optional[str] = None,
                                  create_embeddings: bool = True,
                                  create_contextualized: Optional[bool] = None,  # None means use config default
                                  llm_model_for_context: Optional[str] = None,  # e.g., "gpt-3.5-turbo"
                                  chunk_options: Optional[dict] = None,
                                  hierarchical_chunking: Optional[bool] = None,
                                  hierarchical_template: Optional[dict] = None,
                                  base_metadata: Optional[dict[str, Any]] = None):
        """
        Processes content by chunking, optionally contextualizing, generating embeddings,
        and storing them in ChromaDB and references in SQL DB.

        Args:
            content: Raw document text to process.
            media_id: Logical identifier for the document.
            file_name: File name associated with the content.
            collection_name: Optional Chroma collection name.
            embedding_model_id_override: Optional embedding model override.
            create_embeddings: Whether to generate embeddings.
            create_contextualized: Whether to generate contextual summaries.
            llm_model_for_context: Optional LLM for contextualization.
            chunk_options: Chunking configuration.
            hierarchical_chunking: Enable hierarchical chunking.
            hierarchical_template: Template for hierarchical chunking.
            base_metadata: Optional base metadata to merge into each chunk's metadata
                           before storage (e.g., domain-specific identifiers).
        """
        target_collection = self.get_or_create_collection(collection_name)

        current_op_embedding_model_id = embedding_model_id_override or self.default_embedding_model_id
        if not current_op_embedding_model_id and create_embeddings:
            logger.error(
                f"User '{self.user_id}': No embedding model ID (default or override) for media_id {media_id}. Cannot create embeddings.")
            raise ValueError("Embedding model ID not specified for content processing with embeddings.")

        # Read contextual settings from config if not explicitly provided
        if create_contextualized is None:
            # Try to get from embedding_config first, then fall back to False
            create_contextualized = self.embedding_config.get("enable_contextual_chunking", False)
            logger.debug(f"Using contextual chunking from config: {create_contextualized}")

        effective_llm_model_for_context = llm_model_for_context or self.embedding_config.get(
            "contextual_llm_model", self.embedding_config.get(
                "default_llm_for_contextualization", "gpt-3.5-turbo"))
        # Determine provider for contextualization
        try:
            from tldw_Server_API.app.core.config import settings as _settings  # type: ignore
        except _CHROMA_NONCRITICAL_EXCEPTIONS:
            _settings = {}
        ctx_provider = (
            self.embedding_config.get("contextual_llm_provider")
            or str(_settings.get("default_api", "openai"))
        )

        logger.info(
            f"User '{self.user_id}': Processing content for media_id {media_id} "
            f"in collection '{target_collection.name}' using embedding_model_id '{current_op_embedding_model_id or 'N/A'}'. "
            f"Contextualization: {create_contextualized} with LLM '{effective_llm_model_for_context if create_contextualized else 'N/A'}'."
        )
        try:
            # Chunking (pass options as kwargs for compatibility). Support hierarchical mode.
            effective_chunk_opts: dict = dict(chunk_options or {})
            # Enable adaptive chunking with overlap tuning by default for large docs
            effective_chunk_opts.setdefault('adaptive', True)
            effective_chunk_opts.setdefault('adaptive_overlap', True)
            if hierarchical_chunking is True or (hierarchical_template and isinstance(hierarchical_template, dict)):
                effective_chunk_opts['hierarchical'] = True if hierarchical_chunking is None else bool(hierarchical_chunking)
                if hierarchical_template:
                    effective_chunk_opts['hierarchical_template'] = hierarchical_template
                # Default to sentences for better boundaries unless explicitly provided
                effective_chunk_opts.setdefault('method', 'sentences')
            chunks = chunk_for_embedding(content, file_name, **effective_chunk_opts)
            if not chunks:
                logger.warning(
                    f"User '{self.user_id}': No chunks generated for media_id {media_id}, file {file_name}. Skipping storage.")
                return

            # Ingest-time deduplication (near-duplicate removal)
            try:
                from tldw_Server_API.app.core.config import settings as _settings
            except _CHROMA_NONCRITICAL_EXCEPTIONS:
                _settings = {}
            try:
                ingest_dedup_enabled = bool(_settings.get("INGEST_ENABLE_DEDUP", True))
                dedup_threshold = float(_settings.get("INGEST_DEDUP_THRESHOLD", 0.9))
            except _CHROMA_NONCRITICAL_EXCEPTIONS:
                ingest_dedup_enabled = True
                dedup_threshold = 0.9
            duplicate_map: dict[str, str] = {}
            if ingest_dedup_enabled and chunks and len(chunks) > 1:
                chunks, duplicate_map = self._dedupe_text_chunks(chunks, threshold=dedup_threshold)
                if duplicate_map:
                    logger.info(f"User '{self.user_id}': Deduplicated {len(duplicate_map)} near-duplicate chunks for media_id {media_id}.")

            # TODO: Point 6 - MediaDatabase interaction
            # Placeholder for new MediaDatabase interactions:
            # sql_db_chunks_to_add = []
            # for i, chunk_info in enumerate(chunks):
            #     sql_db_chunks_to_add.append({
            #         "text": chunk_info['text'],
            #         "start_index": chunk_info['metadata'].get('start_index'),
            #         "end_index": chunk_info['metadata'].get('end_index'),
            #         # ... other fields for the new MediaDatabase ...
            #     })
            # if sql_db_chunks_to_add:
            #     # media_db_instance.add_media_chunks_in_batches(media_id=media_id, chunks_to_add=sql_db_chunks_to_add)
            #     logger.info(f"User '{self.user_id}': TODO - Stored {len(sql_db_chunks_to_add)} chunk references in SQL DB for media_id {media_id}.")
            # Optional: Ingestion-time claims (factual statements)
            try:
                from tldw_Server_API.app.core.config import settings as _settings
                if bool(_settings.get("ENABLE_INGESTION_CLAIMS", False)):
                    # Only proceed if we have a DB path attached to this manager
                    db_path = getattr(self, "db_path", None)
                    if db_path:
                        from tldw_Server_API.app.core.Claims_Extraction.ingestion_claims import (
                            extract_claims_for_chunks,
                            store_claims,
                        )
                        # Build map: chunk_index -> chunk_text
                        chunk_text_map = {}
                        for ch in chunks:
                            meta = ch.get("metadata", {}) or {}
                            idx = int(meta.get("chunk_index") or meta.get("index") or 0)
                            chunk_text_map[idx] = ch.get("text") or ch.get("content") or ""

                        max_per = int(_settings.get("CLAIMS_MAX_PER_CHUNK", 3))
                        mode = str(_settings.get("CLAIM_EXTRACTOR_MODE", "heuristic"))
                        claims = extract_claims_for_chunks(chunks, extractor_mode=mode, max_per_chunk=max_per)
                        if claims:
                            try:
                                # Ensure media_id can be used as integer FK
                                mid = int(media_id)
                            except _CHROMA_NONCRITICAL_EXCEPTIONS:
                                logger.warning(f"Claims disabled for non-integer media_id: {media_id}")
                            else:
                                with managed_media_database(
                                    str(_settings.get("SERVER_CLIENT_ID", "SERVER_API_V1")),
                                    db_path=str(db_path),
                                    initialize=False,
                                    suppress_close_exceptions=_CHROMA_NONCRITICAL_EXCEPTIONS,
                                ) as db:
                                    inserted = store_claims(
                                        db,
                                        media_id=mid,
                                        chunk_texts_by_index=chunk_text_map,
                                        claims=claims,
                                        extractor=mode,
                                        extractor_version="v1",
                                    )
                                logger.info(f"User '{self.user_id}': Stored {inserted} ingestion-time claims for media {mid}.")
                        # Optional: embed claims into a dedicated Chroma collection
                        try:
                            if bool(_settings.get("CLAIMS_EMBED", False)) and claims:
                                claim_texts = [c.get("claim_text", "") for c in claims if c.get("claim_text")]
                                if claim_texts:
                                    # Prefer claims-specific model, then request override, then default.
                                    configured_claim_model = str(_settings.get("CLAIMS_EMBED_MODEL_ID", "") or "").strip()
                                    claim_model_id = configured_claim_model or embedding_model_id_override or self.default_embedding_model_id
                                    emb_vectors = create_embeddings_batch(
                                        texts=claim_texts,
                                        user_app_config=self.user_embedding_config,
                                        model_id_override=claim_model_id
                                    )
                                    # Create or access claims collection
                                    coll_name = f"claims_for_{self.user_id}"
                                    try:
                                        claims_coll = self.client.get_or_create_collection(name=coll_name)
                                    except _CHROMA_NONCRITICAL_EXCEPTIONS:
                                        # Fallback via our helper
                                        claims_coll = self.get_or_create_collection(coll_name)
                                    import hashlib as _hashlib
                                    ids = [
                                        f"claim_{media_id}_{(c.get('chunk_index') or 0)}_{_hashlib.sha1(str(c.get('claim_text','')).encode(), usedforsecurity=False).hexdigest()[:12]}"
                                        for c in claims
                                    ]
                                    metas = []
                                    for c in claims:
                                        metas.append({
                                            "media_id": str(media_id),
                                            "chunk_index": int(c.get("chunk_index", 0)),
                                            "source": "claim",
                                            "extractor": mode,
                                            "file_name": str(file_name),
                                        })
                                    claims_coll.upsert(documents=claim_texts, embeddings=emb_vectors, ids=ids, metadatas=metas)
                                    logger.info(f"User '{self.user_id}': Upserted {len(ids)} claim embeddings to collection '{coll_name}'.")
                        except _CHROMA_NONCRITICAL_EXCEPTIONS as e_emb:
                            logger.debug(f"Claim embedding step skipped/failed: {e_emb}")
                    else:
                        logger.debug("Ingestion-time claims enabled but no db_path attached to ChromaDBManager; skipping.")
            except _CHROMA_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"User '{self.user_id}': Ingestion-time claims step failed (non-fatal): {e}")
            # End TODO MediaDatabase

            # Determine strategy, budgets, and window size defaults
            # Strategy: 'auto' (default), 'full', 'window', 'outline_window'
            context_strategy = 'auto'
            token_budget: int = 6000
            context_window_size: Optional[int] = None
            store_context_header_in_docs: bool = False
            prepend_header_to_embedding: bool = True
            if isinstance(chunk_options, dict):
                try:
                    cws = chunk_options.get("context_window_size")
                    if isinstance(cws, str):
                        cws = int(cws)
                    if isinstance(cws, int) and cws > 0:
                        context_window_size = cws
                except _CHROMA_NONCRITICAL_EXCEPTIONS:
                    context_window_size = None
                # Optional toggles to control header usage
                sch = chunk_options.get("store_contextual_header_in_docs")
                if isinstance(sch, bool):
                    store_context_header_in_docs = sch
                phe = chunk_options.get("prepend_header_to_embedding")
                if isinstance(phe, bool):
                    prepend_header_to_embedding = phe
                # Strategy and budget from request
                strat = chunk_options.get("context_strategy")
                if isinstance(strat, str):
                    context_strategy = strat.lower().strip() or context_strategy
                tb = chunk_options.get("context_token_budget")
                try:
                    if isinstance(tb, str):
                        tb = int(tb)
                    if isinstance(tb, int) and tb > 0:
                        token_budget = tb
                except _CHROMA_NONCRITICAL_EXCEPTIONS:
                    pass

            # Fall back to embedding_config defaults if provided
            try:
                emb_cfg = self.embedding_config or {}
                strat_cfg = emb_cfg.get("context_strategy")
                if isinstance(strat_cfg, str) and context_strategy == 'auto':
                    context_strategy = strat_cfg.lower().strip() or context_strategy
                tb_cfg = emb_cfg.get("context_token_budget")
                if token_budget == 6000 and isinstance(tb_cfg, int) and tb_cfg > 0:
                    token_budget = tb_cfg
                # Global window size default
                if context_window_size is None:
                    cfg_cws = emb_cfg.get("context_window_size", None)
                    if isinstance(cfg_cws, int) and cfg_cws > 0:
                        context_window_size = cfg_cws
                # Detect explicit global full-doc lock-in: key present with None value
                force_full_doc_context = ('context_window_size' in emb_cfg and emb_cfg.get('context_window_size') is None)
            except _CHROMA_NONCRITICAL_EXCEPTIONS:
                force_full_doc_context = False

            # Decide whether to use full document or a window per chunk
            doc_tokens = self._estimate_token_count(content)
            use_full_doc_context = True
            include_outline = False
            if force_full_doc_context and context_strategy in ('auto', 'full'):
                use_full_doc_context = True
                include_outline = False
            else:
                if context_strategy == 'full':
                    use_full_doc_context = True
                elif context_strategy == 'window':
                    use_full_doc_context = False
                elif context_strategy == 'outline_window':
                    use_full_doc_context = False
                    include_outline = True
                else:  # auto
                    if doc_tokens <= token_budget:
                        use_full_doc_context = True
                    else:
                        use_full_doc_context = False
                        include_outline = True  # auto: add outline when falling back to window

            # Build document outline once if needed
            doc_outline = ""
            if create_contextualized and not use_full_doc_context and include_outline:
                # Resolve contextualization temperature from embedding config
                ctx_temp = None
                try:
                    tval = emb_cfg.get("contextual_llm_temperature")
                    if tval is not None:
                        ctx_temp = float(tval)
                except _CHROMA_NONCRITICAL_EXCEPTIONS:
                    ctx_temp = None
                doc_outline = self._build_document_outline(
                    ctx_provider,
                    effective_llm_model_for_context,
                    content,
                    ctx_temp,
                )

            if create_embeddings:
                docs_for_chroma = []  # This will hold the text that's actually stored in Chroma
                texts_for_embedding_generation = []  # This will hold the text used to generate embeddings

                for chunk in chunks:
                    chunk_text = chunk['text']
                    docs_for_chroma.append(chunk_text)  # Store original chunk text in Chroma document

                    if create_contextualized:
                        # Compute document context per chosen strategy
                        meta = chunk.get('metadata', {}) or {}
                        start_idx = meta.get('start_char', meta.get('start_index'))
                        end_idx = meta.get('end_char', meta.get('end_index'))
                        # Default window size if needed
                        default_window_chars = 1200
                        effective_cws = context_window_size if isinstance(context_window_size, int) and context_window_size > 0 else default_window_chars

                        if use_full_doc_context:
                            combined_context_doc = content
                        else:
                            # Build a window slice
                            windowed_doc = content
                            if isinstance(start_idx, (int, float)) and isinstance(end_idx, (int, float)):
                                try:
                                    s = max(0, int(start_idx) - int(effective_cws))
                                    e = min(len(content), int(end_idx) + int(effective_cws))
                                    windowed_doc = content[s:e]
                                    meta['context_window_start'] = s
                                    meta['context_window_end'] = e
                                    meta['context_window_size_used'] = int(effective_cws)
                                except _CHROMA_NONCRITICAL_EXCEPTIONS:
                                    windowed_doc = content
                            # Combine outline with window when available
                            if include_outline and doc_outline:
                                combined_context_doc = f"Outline:\n{doc_outline}\n\nWindow:\n{windowed_doc}"
                            else:
                                combined_context_doc = windowed_doc

                        # Resolve contextualization temperature (chunk-scope)
                        ctx_temp = None
                        try:
                            tval = self.embedding_config.get("contextual_llm_temperature")
                            if tval is not None:
                                ctx_temp = float(tval)
                        except _CHROMA_NONCRITICAL_EXCEPTIONS:
                            ctx_temp = None

                        context_summary = self.situate_context(
                            ctx_provider,
                            effective_llm_model_for_context,
                            combined_context_doc,
                            chunk_text,
                            ctx_temp,
                        )

                        # Build an AutoContext-style header for downstream reuse
                        section_name = meta.get('section') or meta.get('chapter_title') or meta.get('header')
                        safe_section = section_name if isinstance(section_name, str) and section_name.strip() else "Unknown"
                        context_header = f"Doc: {file_name} | Section: {safe_section} | Summary: {context_summary}"

                        # Persist header and summary reference in metadata
                        meta['context_header'] = context_header
                        meta['contextual_summary_ref'] = context_summary

                        # Build text to embed; keep the explicit Contextual Summary marker for compatibility
                        if prepend_header_to_embedding:
                            text_to_embed = f"{context_header}\n\n{chunk_text}\n\nContextual Summary: {context_summary}"
                        else:
                            text_to_embed = f"{chunk_text}\n\nContextual Summary: {context_summary}"

                        # Optionally store header inline with the text in Chroma documents
                        if store_context_header_in_docs and docs_for_chroma:
                            docs_for_chroma[-1] = f"{context_header}\n\n{chunk_text}"
                        texts_for_embedding_generation.append(text_to_embed)
                        # If you want to store the contextualized text in Chroma's document field:
                        # docs_for_chroma[-1] = text_to_embed
                    else:
                        texts_for_embedding_generation.append(chunk_text)

                if not texts_for_embedding_generation:
                    logger.warning(
                        f"User '{self.user_id}': No texts prepared for embedding for media_id {media_id}. Skipping embedding creation.")
                else:
                    # TODO: Point 4 - Async/Batching for create_embeddings_batch if it supports async
                    embeddings = create_embeddings_batch(
                        texts=texts_for_embedding_generation,
                        user_app_config=self.user_embedding_config,
                        model_id_override=current_op_embedding_model_id
                    )

                    ids = [f"{media_id}_chunk_{i}" for i in range(len(chunks))]  # 0-indexed chunks

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
                                except _CHROMA_NONCRITICAL_EXCEPTIONS:
                                    return None
                        return None

                    def _normalize_chunk_type(value: Any) -> Optional[str]:
                        try:
                            return Chunker.normalize_chunk_type(value)
                        except _CHROMA_NONCRITICAL_EXCEPTIONS:
                            return None

                    metadatas = []
                    for i, chunk_info in enumerate(chunks):
                        meta = {
                            "media_id": str(media_id),
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "file_name": str(file_name),  # Or chunk_info['metadata']['file_name']
                            "contextualized": create_contextualized,
                            # Store original text for reference even if docs_for_chroma has contextualized text
                            "original_chunk_text_ref": chunk_info['text'][:200] + "..." if len(
                                chunk_info['text']) > 200 else chunk_info['text']
                        }
                        # Add metadata from chunk_for_embedding
                        chunk_meta = chunk_info.get('metadata', {}) or {}
                        meta.update(chunk_meta)

                        raw_chunk_type = (
                            chunk_meta.get("chunk_type")
                            or chunk_info.get("chunk_type")
                            or chunk_meta.get("paragraph_kind")
                            or chunk_meta.get("type")
                            or chunk_meta.get("kind")
                        )
                        normalized_chunk_type = _normalize_chunk_type(raw_chunk_type)
                        if normalized_chunk_type:
                            meta["chunk_type"] = normalized_chunk_type
                        elif "chunk_type" not in meta:
                            meta["chunk_type"] = "text"

                        start_val = (
                            meta.get("start_char")
                            or chunk_meta.get("start_char")
                            or chunk_meta.get("start_index")
                            or chunk_meta.get("start_offset")
                        )
                        end_val = (
                            meta.get("end_char")
                            or chunk_meta.get("end_char")
                            or chunk_meta.get("end_index")
                            or chunk_meta.get("end_offset")
                        )
                        start_int = _coerce_int(start_val)
                        if start_int is not None:
                            meta["start_char"] = start_int
                        end_int = _coerce_int(end_val)
                        if end_int is not None:
                            meta["end_char"] = end_int

                        citation_payload = chunk_meta.get("citation")
                        if not isinstance(citation_payload, dict):
                            citation_payload = {}

                        def _citation_value(key: str, aliases: tuple[str, ...] = (), _citation_payload=citation_payload, _chunk_meta=chunk_meta) -> Any:
                            if key in _citation_payload:
                                return _citation_payload.get(key)
                            for alias in aliases:
                                if alias in _chunk_meta:
                                    return _chunk_meta.get(alias)
                            if key in _chunk_meta:
                                return _chunk_meta.get(key)
                            return None

                        citation_numeric_fields = {
                            "page_number": ("page", "page_no"),
                            "paragraph_number": (),
                            "line_number": (),
                            "slide_number": (),
                            "row_number": (),
                            "column_number": (),
                            "start_timestamp_ms": ("start_ts_ms", "start_ts"),
                            "end_timestamp_ms": ("end_ts_ms", "end_ts"),
                        }
                        for field_key, aliases in citation_numeric_fields.items():
                            dest_key = f"citation.{field_key}"
                            if dest_key in meta:
                                continue
                            value = _citation_value(field_key, aliases)
                            coerced = _coerce_int(value)
                            if coerced is not None:
                                meta[dest_key] = coerced

                        sheet_name = _citation_value("sheet_name")
                        if sheet_name and "citation.sheet_name" not in meta:
                            meta["citation.sheet_name"] = str(sheet_name)

                        # Merge in any base-level metadata provided by caller (base first, then chunk-specific)
                        if base_metadata:
                            try:
                                # Reverse order: start with base-level metadata, then overlay chunk-specific metadata
                                base_copy = dict(base_metadata)
                                base_copy.update(meta)
                                meta = base_copy
                            except TypeError as e:
                                # Best-effort merge; log and skip on type errors
                                logger.debug(
                                    f"User '{self.user_id}': Failed to merge base_metadata for chunk {i} "
                                    f"(media_id: {media_id}). Type error: {e}"
                                )

                        if create_contextualized:
                            # If docs_for_chroma contains original text, but texts_for_embedding_generation has context,
                            # you might want to store the generated context summary in metadata.
                            context_part = texts_for_embedding_generation[i].split("\n\nContextual Summary: ", 1)
                            if len(context_part) > 1:
                                meta["contextual_summary_ref"] = context_part[1]

                        metadatas.append(meta)

                    self.store_in_chroma(
                        collection_name=target_collection.name,
                        texts=docs_for_chroma,  # Text to store in ChromaDB document field
                        embeddings=embeddings,
                        ids=ids,
                        metadatas=metadatas,
                        embedding_model_id_for_dim_check=current_op_embedding_model_id
                    )

            # TODO: Point 6 - MediaDatabase interaction
            # mark_media_as_processed(media_db_instance, media_id)
            # media_db_instance.execute_query(
            # "INSERT OR REPLACE INTO media_fts (rowid, title, content) SELECT id, title, content FROM Media WHERE id = ?",
            # (media_id,),
            # commit=True
            # )
            logger.info(f"User '{self.user_id}': TODO - Mark media {media_id} as processed and update FTS.")
            # End TODO MediaDatabase

            logger.info(f"User '{self.user_id}': Finished processing and storing content for media_id {media_id}")

        except ValueError as ve:  # Catch specific configuration/input errors
            logger.error(f"User '{self.user_id}': Input or configuration error processing media_id {media_id}: {ve}",
                         exc_info=True)
            raise  # Re-raise to signal failure
        except RuntimeError as rte:  # Catch ChromaDB or system-level issues
            logger.error(f"User '{self.user_id}': Runtime error processing media_id {media_id}: {rte}", exc_info=True)
            raise
        except _CHROMA_NONCRITICAL_EXCEPTIONS as e:  # General catch-all
            logger.error(
                f"User '{self.user_id}': Unexpected error in process_and_store_content for media_id {media_id}: {e}",
                exc_info=True)
            raise  # Re-raise for unhandled issues

    def store_in_chroma(self, collection_name: Optional[str], texts: list[str],
                        embeddings: Union[np.ndarray, list[list[float]]],  # Type hint improved
                        ids: list[str], metadatas: list[dict[str, Any]],
                        embedding_model_id_for_dim_check: Optional[str] = None):
        """Stores embeddings and associated data into a ChromaDB collection."""
        if not texts or not ids or not metadatas or embeddings is None or len(
                embeddings) == 0:  # Check embeddings has content
            logger.error(
                f"User '{self.user_id}': Invalid input to store_in_chroma. Texts, ids, metadatas, or embeddings are empty/None.")
            raise ValueError(
                "Texts, ids, metadatas lists must be non-empty, and embeddings must be provided and non-empty.")

        if not (len(texts) == len(embeddings) == len(ids) == len(metadatas)):
            error_msg = (f"Input list length mismatch: Texts({len(texts)}), Embeddings({len(embeddings)}), "
                         f"IDs({len(ids)}), Metadatas({len(metadatas)})")
            logger.error(f"User '{self.user_id}': {error_msg}")
            raise ValueError(error_msg)

        if isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()
        elif isinstance(embeddings, list) and all(isinstance(e, list) for e in embeddings):
            embeddings_list = embeddings
        else:
            logger.error(
                f"User '{self.user_id}': Embeddings type mismatch. Expected List[List[float]] or np.ndarray, got {type(embeddings)}.")
            raise TypeError("Embeddings must be a list of lists (vectors) or a 2D numpy array.")

        if not embeddings_list or not isinstance(embeddings_list[0], list) or len(embeddings_list[0]) == 0:
            logger.error(f"User '{self.user_id}': Embeddings list is empty or malformed after conversion.")
            raise ValueError("No valid embeddings provided after potential conversion.")

        target_collection = self.get_or_create_collection(collection_name)
        new_embedding_dim = len(embeddings_list[0])

        with self._lock:
            logger.info(
                f"User '{self.user_id}': Attempting to store {len(embeddings_list)} embeddings (dim: {new_embedding_dim}) "
                f"in ChromaDB Collection: '{target_collection.name}'.")
            try:
                cleaned_metadatas = [self._clean_metadata(metadata) for metadata in metadatas]

                # Dimension Check and Collection Recreation (if needed)
                # This check is more robust if the collection stores its expected dimension in metadata
                collection_meta = target_collection.metadata
                existing_dim_from_meta = None
                if collection_meta and "embedding_dimension" in collection_meta:
                    existing_dim_from_meta = int(collection_meta["embedding_dimension"])

                if existing_dim_from_meta and existing_dim_from_meta != new_embedding_dim:
                    logger.warning(
                        f"User '{self.user_id}': Embedding dimension mismatch for collection '{target_collection.name}'. "
                        f"Collection expected dim (from metadata): {existing_dim_from_meta}, New: {new_embedding_dim} "
                        f"(from model_id '{embedding_model_id_for_dim_check or 'Unknown'}'). Recreating collection."
                    )
                    self.client.delete_collection(name=target_collection.name)
                    new_coll_meta = {"embedding_dimension": new_embedding_dim}
                    if embedding_model_id_for_dim_check:
                        new_coll_meta["source_model_id"] = embedding_model_id_for_dim_check
                    target_collection = self.client.create_collection(name=target_collection.name,
                                                                      metadata=new_coll_meta)
                elif not existing_dim_from_meta and target_collection.count() > 0:  # Has items but no dim in meta
                    # Fallback: get an existing embedding to check dimension
                    existing_item = target_collection.get(limit=1, include=['embeddings'])
                    embeddings_exist = existing_item.get('embeddings') is not None
                    if embeddings_exist and len(existing_item['embeddings']) > 0:
                        existing_dim_from_sample = len(existing_item['embeddings'][0])
                        if existing_dim_from_sample != new_embedding_dim:
                            logger.warning(
                                f"User '{self.user_id}': Dim mismatch (sampled). Existing: {existing_dim_from_sample}, New: {new_embedding_dim}. Recreating '{target_collection.name}'."
                            )
                            self.client.delete_collection(name=target_collection.name)
                            new_coll_meta = {"embedding_dimension": new_embedding_dim}
                            if embedding_model_id_for_dim_check:
                                new_coll_meta[
                                "source_model_id"] = embedding_model_id_for_dim_check
                            target_collection = self.client.create_collection(name=target_collection.name,
                                                                              metadata=new_coll_meta)

                # Batch upsert for potentially large number of embeddings
                # ChromaDB's upsert handles batching internally, but if we had extremely large lists,
                # we might do it with self._batched here. For now, single upsert is fine.
                target_collection.upsert(
                    documents=texts,
                    embeddings=embeddings_list,
                    ids=ids,
                    metadatas=cleaned_metadatas
                )
                logger.info(
                    f"User '{self.user_id}': Successfully upserted {len(embeddings_list)} items to '{target_collection.name}'.")

            except chromadb.errors.ChromaError as ce:  # Catch specific ChromaDB errors
                if _is_transient_chroma_segment_error(ce):
                    logger.warning(
                        f"User '{self.user_id}': Retrying ChromaDB upsert for collection "
                        f"'{target_collection.name}' after transient segment-reader error: {ce}"
                    )
                    time.sleep(0.05)
                    try:
                        target_collection = self.client.get_or_create_collection(name=target_collection.name)
                        target_collection.upsert(
                            documents=texts,
                            embeddings=embeddings_list,
                            ids=ids,
                            metadatas=cleaned_metadatas
                        )
                        logger.info(
                            f"User '{self.user_id}': Successfully upserted {len(embeddings_list)} "
                            f"items to '{target_collection.name}' after retry.")
                        return target_collection
                    except chromadb.errors.ChromaError as retry_error:
                        logger.error(
                            f"User '{self.user_id}': ChromaDB retry in store_in_chroma for collection "
                            f"'{target_collection.name}' failed: {retry_error}",
                            exc_info=True)
                        raise RuntimeError(
                            f"ChromaDB operation failed after retry: {retry_error}"
                        ) from retry_error
                logger.error(
                    f"User '{self.user_id}': ChromaDB error in store_in_chroma for collection '{target_collection.name}': {ce}",
                    exc_info=True)
                raise RuntimeError(f"ChromaDB operation failed: {ce}") from ce
            except _CHROMA_NONCRITICAL_EXCEPTIONS as e:
                import traceback
                tb = traceback.format_exc()
                logger.error(
                    f"User '{self.user_id}': Unexpected error in store_in_chroma for collection '{target_collection.name}': {e}\nTraceback:\n{tb}")
                raise RuntimeError(f"Unexpected error during ChromaDB storage: {e}") from e
        return target_collection

    def vector_search(self, query: str, collection_name: Optional[str] = None, k: int = 10,
                      embedding_model_id_override: Optional[str] = None,
                      where_filter: Optional[dict[str, Any]] = None,
                      # Use the Literal type for include_fields
                      include_fields: Optional[list[ChromaIncludeLiteral]] = None
                      ) -> list[dict[str, Any]]:
        """Performs a vector search in the specified collection."""
        target_collection = self.get_or_create_collection(collection_name)

        query_embedding_model_id = embedding_model_id_override or self.default_embedding_model_id
        if not query_embedding_model_id:
            logger.error(
                f"User '{self.user_id}': No embedding model ID (default or override) for vector search. Cannot generate query embedding.")
            raise ValueError("Embedding model ID not specified for vector search.")

        # The default value must also conform to List[ChromaIncludeLiteral]
        effective_include_fields: list[ChromaIncludeLiteral]
        if include_fields is None:
            effective_include_fields = ["documents", "metadatas", "distances"]
        else:
            effective_include_fields = include_fields  # Assume caller provides a correctly typed list

        with self._lock:
            try:
                logger.info(
                    f"User '{self.user_id}': Vector search in '{target_collection.name}' for query: '{query[:50]}...' "
                    f"using model_id '{query_embedding_model_id}'. k={k}, Filter: {where_filter is not None}."
                )

                # Corrected call to create_embedding (from a previous iteration)
                query_embedding_single: list[float] = create_embedding(
                    text=query,
                    user_app_config=self.user_embedding_config,  # Pass the main app_config
                    model_id_override=query_embedding_model_id,
                )

                if not query_embedding_single or not isinstance(query_embedding_single, list) or \
                        not all(isinstance(x, (float, int)) for x in query_embedding_single):
                    logger.error(
                        f"User '{self.user_id}': create_embedding did not return a valid List[float] for query '{query[:50]}...'. Got: {type(query_embedding_single)}")
                    if not query_embedding_single:
                        raise ValueError(f"Failed to generate query embedding for query: {query[:50]}...")
                    raise TypeError(f"Query embedding is malformed: {query_embedding_single}")

                query_embedding_list_for_chroma: list[list[float]] = [query_embedding_single]

                if not query_embedding_list_for_chroma or not query_embedding_list_for_chroma[0]:
                    logger.error(
                        f"User '{self.user_id}': Failed to prepare a valid query embedding list for '{query[:50]}...'.")
                    return []

                results: QueryResult = target_collection.query(
                    query_embeddings=query_embedding_list_for_chroma,
                    n_results=k,
                    where=self._clean_metadata(where_filter) if where_filter else None,
                    include=effective_include_fields  # Pass the correctly typed list
                )

                # Process results
                output = []
                if not results or not results.get('ids') or not results['ids'][0]:
                    logger.info(
                        f"User '{self.user_id}': No results found for the query in collection '{target_collection.name}'.")
                    return []

                num_results_for_first_query = len(results['ids'][0])
                for i in range(num_results_for_first_query):
                    item = {}
                    # Helper accessors to avoid ambiguous truthiness (e.g., numpy arrays)
                    docs = results.get('documents')
                    metas = results.get('metadatas')
                    dists = results.get('distances')
                    embs = results.get('embeddings')
                    uris = results.get('uris')
                    data_field = results.get('data')

                    # Only fill fields when requested and present with expected outer/inner lengths
                    if "documents" in effective_include_fields and docs is not None and len(docs) > 0 and len(docs[0]) > 0:
                        item["content"] = docs[0][i]
                    if "metadatas" in effective_include_fields and metas is not None and len(metas) > 0 and len(metas[0]) > 0:
                        item["metadata"] = metas[0][i]
                    if "distances" in effective_include_fields and dists is not None and len(dists) > 0 and len(dists[0]) > 0:
                        item["distance"] = dists[0][i]
                    if "embeddings" in effective_include_fields and embs is not None and len(embs) > 0 and len(embs[0]) > 0:
                        emb_val = embs[0][i]
                        try:
                            # Normalize numpy arrays to plain lists for JSON-serializable output
                            emb_val = emb_val.tolist() if hasattr(emb_val, "tolist") else emb_val
                        except _CHROMA_NONCRITICAL_EXCEPTIONS:
                            pass
                        item["embedding"] = emb_val
                    if "uris" in effective_include_fields and uris is not None and len(uris) > 0 and len(uris[0]) > 0:
                        item["uri"] = uris[0][i]
                    if "data" in effective_include_fields and data_field is not None and len(data_field) > 0 and len(data_field[0]) > 0:
                        item["data"] = data_field[0][i]

                    # IDs are generally always included by ChromaDB if results are found
                    if results.get('ids') and results['ids'][0]:
                        item["id"] = results['ids'][0][i]
                    else:  # Should not happen if num_results_for_first_query > 0
                        logger.warning(
                            f"User '{self.user_id}': Missing 'ids' in results despite having num_results > 0. Index: {i}")
                        continue  # Skip this potentially incomplete result item

                    output.append(item)

                logger.info(
                    f"User '{self.user_id}': Found {len(output)} results for query in '{target_collection.name}'.")
                return output

            except ValueError as ve:
                logger.error(
                    f"User '{self.user_id}': Value error during vector search in '{target_collection.name}': {ve}",
                    exc_info=True)
                raise
            # Use specific ChromaError imports if they work
            except _CHROMA_NONCRITICAL_EXCEPTIONS as e:  # More general catch
                error_str = str(e).lower()
                # Try to identify if it's a Chroma-related "collection not found"
                # This is more heuristic and less reliable than specific exception types
                is_chroma_not_found = (
                        type(e).__module__.startswith('chromadb.') and  # Check if error originates from chromadb
                        "collection" in error_str and
                        ("not found" in error_str or "does not exist" in error_str)
                )

                if is_chroma_not_found:
                    # For vector_search:
                    logger.warning(
                        f"User '{self.user_id}': Collection '{target_collection.name}' not found during search: {e}")
                    return []
                    # For delete_collection:
                    # logger.warning(f"User '{self.user_id}': Collection '{collection_name}' not found for deletion: {e}")
                    # # Potentially do not raise
                elif type(e).__module__.startswith('chromadb.'):  # Other Chroma-originated error
                    logger.error(f"User '{self.user_id}': ChromaDB-related error: {e}", exc_info=True)
                    raise RuntimeError(f"ChromaDB operation failed: {e}") from e
                else:  # Other unexpected error
                    logger.error(f"User '{self.user_id}': Unexpected error: {e}", exc_info=True)
                    raise RuntimeError(f"Unexpected error during operation: {e}") from e

    def reset_chroma_collection(self, collection_name: Optional[str] = None):
        """Resets (deletes and recreates) a ChromaDB collection."""
        name_to_reset = collection_name or self.get_user_default_collection_name()
        with self._lock:
            try:
                logger.info(f"User '{self.user_id}': Attempting to reset ChromaDB collection: '{name_to_reset}'.")
                self.client.delete_collection(name=name_to_reset)
                # No specific metadata needed on basic recreate, store_in_chroma will handle dim metadata
                self.client.create_collection(name=name_to_reset)
                logger.info(f"User '{self.user_id}': Successfully reset ChromaDB collection: '{name_to_reset}'.")
            except chromadb.errors.ChromaError as ce:
                # If deleting a non-existent collection, Chroma might error.
                # We still want to ensure it's created.
                if "does not exist" in str(ce).lower():  # Check if it's an error about non-existence
                    logger.warning(
                        f"User '{self.user_id}': Collection '{name_to_reset}' did not exist during delete for reset. Will attempt creation.")
                else:  # Other Chroma error during delete
                    logger.error(
                        f"User '{self.user_id}': ChromaDB error deleting collection '{name_to_reset}' during reset: {ce}",
                        exc_info=True)
                    # Decide if we should still try to create or raise
                try:
                    self.client.create_collection(name=name_to_reset)  # Attempt creation
                    logger.info(
                        f"User '{self.user_id}': Created ChromaDB collection '{name_to_reset}' after (failed) delete attempt.")
                except _CHROMA_NONCRITICAL_EXCEPTIONS as ice:  # Inner create exception
                    logger.error(
                        f"User '{self.user_id}': Failed to create collection '{name_to_reset}' after reset attempt: {ice}",
                        exc_info=True)
                    raise RuntimeError(f"Failed to finalize reset for collection '{name_to_reset}': {ice}") from ice
            except _CHROMA_NONCRITICAL_EXCEPTIONS as e:  # Catch other errors during delete
                logger.error(f"User '{self.user_id}': Unexpected error resetting collection '{name_to_reset}': {e}",
                             exc_info=True)
                raise RuntimeError(f"Unexpected error during collection reset: {e}") from e

    def delete_from_collection(self, ids: list[str], collection_name: Optional[str] = None):
        """Deletes items from a collection by their IDs."""
        if not ids:
            logger.warning(f"User '{self.user_id}': No IDs provided for deletion. Skipping.")
            return

        target_collection = self.get_or_create_collection(collection_name)
        with self._lock:
            try:
                logger.info(
                    f"User '{self.user_id}': Deleting {len(ids)} item(s) from collection '{target_collection.name}'.")
                target_collection.delete(ids=ids)
            except chromadb.errors.ChromaError as ce:
                logger.error(
                    f"User '{self.user_id}': ChromaDB error deleting IDs {ids} from collection "
                    f"'{target_collection.name}': {ce}", exc_info=True)
                raise RuntimeError(f"ChromaDB operation failed: {ce}") from ce
            except _CHROMA_NONCRITICAL_EXCEPTIONS as e:
                logger.error(
                    f"User '{self.user_id}': Unexpected error deleting IDs {ids} from collection "
                    f"'{target_collection.name}': {e}", exc_info=True)
                raise RuntimeError(f"Unexpected error during deletion: {e}") from e

    # --- Ingest-time utilities ---
    def _dedupe_text_chunks(self, chunks: list[dict[str, Any]], threshold: float = 0.9) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Remove near-duplicate chunks using Jaccard + SimHash (short-text friendly).

        Args:
            chunks: List of {'text': str, 'metadata': {...}} dicts
            threshold: Jaccard similarity >= threshold marks as duplicate

        Returns:
            (filtered_chunks, duplicate_map) where duplicate_map maps duplicate_chunk_uid -> canonical_chunk_uid
        """
        try:
            k = 7  # shingle size

            def _simhash64(s: str) -> int:
                """Compute a simple 64-bit SimHash for short technical text.

                Uses word-level hashing with weighted character n-grams to improve
                discrimination on short inputs. Deterministic and fast; not cryptographic.
                """
                try:
                    import hashlib as _hash
                    words = [w for w in re.split(r"\W+", s.lower()) if w]
                    if not words:
                        return 0
                    # small char-gram features per word to reduce collisions
                    def grams(w: str) -> list[str]:
                        g = set()
                        w2 = f"^{w}$"
                        for n in (2, 3):
                            if len(w2) >= n:
                                for i in range(len(w2) - n + 1):
                                    g.add(w2[i:i+n])
                        return list(g) or [w]
                    bits = [0] * 64
                    for w in words:
                        feats = grams(w)
                        for f in feats:
                            h = int(_hash.sha1(f.encode("utf-8"), usedforsecurity=False).hexdigest()[:16], 16)
                            for b in range(64):
                                if (h >> b) & 1:
                                    bits[b] += 1
                                else:
                                    bits[b] -= 1
                    out = 0
                    for b in range(64):
                        if bits[b] >= 0:
                            out |= (1 << b)
                    return out
                except _CHROMA_NONCRITICAL_EXCEPTIONS:
                    return 0

            def _hamming(a: int, b: int) -> int:
                x = a ^ b
                # Kernighan popcount
                c = 0
                while x:
                    x &= x - 1
                    c += 1
                return c
            def shingles(s: str) -> set[str]:
                s = s or ""
                if len(s) < k:
                    return {s}
                return {s[i:i+k] for i in range(0, len(s) - k + 1)}

            canon: list[tuple[str, set[str], int]] = []  # (uid, shingle_set, simhash)
            duplicate_map: dict[str, str] = {}
            filtered: list[dict[str, Any]] = []
            # SimHash gating config
            try:
                from tldw_Server_API.app.core.config import settings as _settings
            except _CHROMA_NONCRITICAL_EXCEPTIONS:
                _settings = {}
            use_simhash = bool(_settings.get("INGEST_DEDUP_USE_SIMHASH", True))
            simhash_hamming_thresh = int(_settings.get("INGEST_SIMHASH_HAMMING_THRESHOLD", 3))

            for ch in chunks:
                txt = ch.get('text') or ''
                md = ch.get('metadata') or {}
                uid = str(md.get('chunk_uid') or md.get('chunk_index') or f"idx_{len(filtered)}")
                sh = shingles(txt)
                sh_val = _simhash64(txt) if use_simhash else 0
                is_dup = False
                for (cuid, cset, csim) in canon:
                    inter = len(sh & cset)
                    union = max(1, len(sh | cset))
                    jac = inter / union
                    if jac >= threshold:
                        # Mark duplicate of canonical cuid
                        duplicate_map[uid] = cuid
                        is_dup = True
                        break
                    if use_simhash and sh_val and csim:
                        if _hamming(sh_val, csim) <= simhash_hamming_thresh:
                            duplicate_map[uid] = cuid
                            is_dup = True
                            break
                if not is_dup:
                    canon.append((uid, sh, sh_val))
                    filtered.append(ch)
            # Annotate duplicates in metadata for traceability
            if duplicate_map:
                for ch in chunks:
                    md = ch.get('metadata') or {}
                    uid = str(md.get('chunk_uid') or md.get('chunk_index') or '')
                    if uid in duplicate_map:
                        md['duplicate_of'] = duplicate_map[uid]
                        ch['metadata'] = md
            return filtered, duplicate_map
        except _CHROMA_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"User '{self.user_id}': Dedupe failed ({e}); returning original chunks.")
            return chunks, {}

    def query_collection_with_precomputed_embeddings(
            self, query_embeddings: list[list[float]],
            n_results: int = 5,
            where_clause: Optional[dict[str, Any]] = None,
            collection_name: Optional[str] = None,
            # Use the Literal type for include_fields
            include_fields: Optional[list[ChromaIncludeLiteral]] = None
    ) -> QueryResult:
        """
        Queries a collection using pre-computed embeddings.
        Args:
            query_embeddings (List[List[float]]): A list of query embedding vectors.
            include_fields (Optional[List[ChromaIncludeLiteral]]): A list of fields to include in the results.
        """
        target_collection = self.get_or_create_collection(collection_name)

        # The default value must also conform to List[ChromaIncludeLiteral]
        effective_include_fields: list[ChromaIncludeLiteral]
        effective_include_fields = ["documents", "metadatas", "distances"] if include_fields is None else include_fields

        with self._lock:
            try:
                if not query_embeddings or not query_embeddings[0]:  # Check the first embedding vector exists
                    logger.error(
                        f"User '{self.user_id}': Empty or malformed query_embeddings provided to query_collection_with_precomputed_embeddings.")
                    raise ValueError("Query embeddings cannot be empty and must contain valid vectors.")

                # Ensure all sub-lists (vectors) in query_embeddings are not empty
                if any(not vec for vec in query_embeddings):
                    logger.error(
                        f"User '{self.user_id}': One or more embedding vectors in query_embeddings is empty.")
                    raise ValueError("All embedding vectors in query_embeddings must be non-empty.")

                return target_collection.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                    where=self._clean_metadata(where_clause) if where_clause else None,
                    include=effective_include_fields  # Pass the correctly typed list
                )
            # Use the more specific ChromaError imports if they work for your version
            except ChromaError as ce:
                if collection_name and _is_transient_chroma_segment_error(ce):
                    logger.warning(
                        f"User '{self.user_id}': Retrying ChromaDB query for collection "
                        f"'{target_collection.name}' after transient segment-reader error: {ce}"
                    )
                    time.sleep(0.05)
                    try:
                        target_collection = self.client.get_or_create_collection(name=collection_name)
                        return target_collection.query(
                            query_embeddings=query_embeddings,
                            n_results=n_results,
                            where=self._clean_metadata(where_clause) if where_clause else None,
                            include=effective_include_fields
                        )
                    except ChromaError as retry_error:
                        logger.error(
                            f"User '{self.user_id}': ChromaDB retry querying collection "
                            f"'{target_collection.name}' failed: {retry_error}",
                            exc_info=True)
                        raise RuntimeError(
                            f"ChromaDB query with precomputed embeddings failed: {retry_error}"
                        ) from retry_error
                logger.error(
                    f"User '{self.user_id}': ChromaDB error querying collection '{target_collection.name}': {ce}",
                    exc_info=True)
                raise RuntimeError(f"ChromaDB query with precomputed embeddings failed: {ce}") from ce
            except ValueError as ve:  # Catch ValueErrors from our checks
                logger.error(
                    f"User '{self.user_id}': Input validation error in query_collection_with_precomputed_embeddings: {ve}",
                    exc_info=True)
                raise  # Re-raise the ValueError
            except _CHROMA_NONCRITICAL_EXCEPTIONS as e:
                logger.error(
                    f"User '{self.user_id}': Unexpected error querying '{target_collection.name}' with precomputed embeddings: {e}",
                    exc_info=True)
                raise RuntimeError(f"Unexpected error during query with precomputed embeddings: {e}") from e

    def count_items_in_collection(self, collection_name: Optional[str] = None) -> int:
        """Counts the number of items in a collection."""
        target_collection = self.get_or_create_collection(collection_name)
        with self._lock:
            try:
                return target_collection.count()
            except _CHROMA_NONCRITICAL_EXCEPTIONS as e:
                logger.error(
                    f"User '{self.user_id}': Error counting items in collection '{target_collection.name}': {e}",
                    exc_info=True)
                # Depending on severity, either return 0 or raise
                raise RuntimeError(f"Failed to count items in collection: {e}") from e

    def get_collection(self, collection_name: str) -> Collection:
        """
        Gets an existing ChromaDB collection without creating one.

        Args:
            collection_name: Name of the collection to retrieve.

        Returns:
            Collection: The ChromaDB collection object.

        Raises:
            KeyError: If the collection does not exist.
            RuntimeError: If the lookup fails for other reasons.
        """
        if not collection_name:
            raise ValueError("collection_name must be provided.")
        with self._lock:
            try:
                return self.client.get_collection(name=collection_name)
            except (KeyError, ValueError) as e:
                raise KeyError(f"Collection '{collection_name}' does not exist") from e
            except _CHROMA_NONCRITICAL_EXCEPTIONS as e:
                logger.error(
                    f"User '{self.user_id}': Failed to get collection '{collection_name}': {e}",
                    exc_info=True,
                )
                raise RuntimeError(f"Failed to get collection '{collection_name}': {e}") from e

    def list_collections(self) -> Sequence[Collection]:
        """Lists all collections for the current user's client."""
        with self._lock:
            try:
                return self.client.list_collections()
            except _CHROMA_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"User '{self.user_id}': Error listing collections: {e}", exc_info=True)
                raise RuntimeError(f"Failed to list collections: {e}") from e

    def delete_collection(self, collection_name: str):
        """Deletes a specific collection by name."""
        if not collection_name:
            raise ValueError("collection_name must be provided for deletion.")
        with self._lock:
            try:
                self.client.delete_collection(name=collection_name)
                logger.info(f"User '{self.user_id}': Successfully deleted collection '{collection_name}'.")
            except chromadb.errors.ChromaError as ce:
                # Handle if collection doesn't exist gracefully or re-raise
                logger.warning(
                    f"User '{self.user_id}': ChromaDB error deleting collection '{collection_name}' (it might not exist): {ce}")
                # Depending on desired strictness, you might not re-raise if it's "does not exist"
                if "does not exist" not in str(ce).lower():
                    raise RuntimeError(f"ChromaDB failed to delete collection '{collection_name}': {ce}") from ce
            except _CHROMA_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"User '{self.user_id}': Unexpected error deleting collection '{collection_name}': {e}",
                             exc_info=True)
                raise RuntimeError(f"Unexpected error deleting collection '{collection_name}': {e}") from e

# Example of how you might instantiate and use it (outside this file):
# from tldw_Server_API.app.core.config import settings
# from tldw_Server_API.app.core.Vector_DB_Management.ChromaDB_Library import ChromaDBManager

# current_user_id = "user123" # Get this from auth context
# chroma_manager = ChromaDBManager(user_id=current_user_id, settings=settings)
# results = chroma_manager.vector_search(query="some query text", k=5)
# chroma_manager.store_in_chroma(texts=["..."], embeddings=[[...]], ids=["..."], metadatas=[{...}])
#
# End of Functions for ChromaDB
#######################################################################################################################

# Compatibility layer for legacy code expecting module-level functions
# This creates a default instance for single-user mode or tests
_default_chroma_manager = None
_manager_lock = threading.Lock()
_TEST_FALLBACK_DIRS: dict[str, Path] = {}
_TEST_STUB_CLIENTS: dict[str, Any] = {}
_TEST_STUB_CLIENTS_LOCK = threading.Lock()
_PERSISTENT_CLIENTS: dict[str, _PersistentClientState] = {}
_PERSISTENT_CLIENTS_LOCK = threading.Lock()


# --------------------
# Test-mode in-memory fallback client (minimal Chroma-like API)
# --------------------

class _InMemorySystem:
    def stop(self):
        return None


class _InMemoryCollection:
    def __init__(self, name: str, metadata: Optional[dict[str, Any]] = None):
        self.name = name
        self._metadata: dict[str, Any] = dict(metadata or {})
        self._docs: dict[str, str] = {}
        self._embs: dict[str, list[float]] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._deleted: bool = False

    def _ensure_active(self) -> None:
        if self._deleted:
            raise RuntimeError(f"Collection '{self.name}' no longer exists")

    @property
    def metadata(self) -> dict[str, Any]:
        self._ensure_active()
        return self._metadata

    def modify(self, metadata: Optional[dict[str, Any]] = None):
        self._ensure_active()
        if metadata:
            self._metadata.update(metadata)

    def count(self) -> int:
        self._ensure_active()
        return len(self._docs)

    def add(self, documents: list[str], embeddings: list[list[float]], ids: list[str], metadatas: Optional[list[dict[str, Any]]] = None):
        return self.upsert(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)

    def upsert(self, documents: list[str], embeddings: list[list[float]], ids: list[str], metadatas: Optional[list[dict[str, Any]]] = None):
        self._ensure_active()
        mds = metadatas or [{} for _ in ids]
        for i, id_ in enumerate(ids):
            self._docs[id_] = documents[i] if documents else ""
            self._embs[id_] = list(embeddings[i]) if embeddings else []
            self._meta[id_] = dict(mds[i] or {})
        return None

    def delete(self, ids: Optional[list[str]] = None, where: Optional[dict[str, Any]] = None):
        self._ensure_active()
        if ids:
            for id_ in list(ids):
                self._docs.pop(id_, None)
                self._embs.pop(id_, None)
                self._meta.pop(id_, None)
        elif where:
            # Delete by metadata filter
            def _match(md: dict[str, Any]) -> bool:
                return all(md.get(k) == v for k, v in (where or {}).items())
            to_delete = [k for k, md in list(self._meta.items()) if _match(md)]
            for id_ in to_delete:
                self._docs.pop(id_, None)
                self._embs.pop(id_, None)
                self._meta.pop(id_, None)
        return None

    def get(self, ids: Optional[list[str]] = None, where: Optional[dict[str, Any]] = None, limit: Optional[int] = None, offset: Optional[int] = None, include: Optional[list[str]] = None) -> dict[str, Any]:
        self._ensure_active()
        # Mirror Chroma's default behavior: include documents and metadatas when include is not specified
        if include is None:
            include = ["documents", "metadatas"]
        keys = []
        keys = [k for k in ids if k in self._docs] if ids else list(self._docs.keys())
        if where:
            def match(m: dict[str, Any]) -> bool:
                return all(m.get(k) == v for k, v in where.items())
            keys = [k for k in keys if match(self._meta.get(k, {}))]
        if offset is not None and offset > 0:
            try:
                keys = keys[offset:]
            except _CHROMA_NONCRITICAL_EXCEPTIONS:
                keys = keys
        if limit is not None:
            keys = keys[:limit]
        out: dict[str, Any] = {"ids": keys}
        if "documents" in include:
            out["documents"] = [self._docs[k] for k in keys]
        if "metadatas" in include:
            out["metadatas"] = [self._meta.get(k, {}) for k in keys]
        if "embeddings" in include:
            out["embeddings"] = [self._embs.get(k, []) for k in keys]
        if "distances" in include:
            out["distances"] = [0.0 for _ in keys]
        return out

    def query(self, query_embeddings: list[list[float]], n_results: int = 10, where: Optional[dict[str, Any]] = None, include: Optional[list[str]] = None) -> dict[str, Any]:
        self._ensure_active()
        # Mirror common defaults: when include is omitted, return documents, metadatas, and distances
        if include is None:
            include = ["documents", "metadatas", "distances"]
        q = query_embeddings[0] if query_embeddings else []
        def match(m: dict[str, Any]) -> bool:
            if not where:
                return True
            return all(m.get(k) == v for k, v in where.items())
        items = []
        for k in self._docs:
            if not match(self._meta.get(k, {})):
                continue
            v = self._embs.get(k, [])
            # Simple cosine-like (fallback to euclidean if zero vector)
            dist = 0.0
            if q and v and len(q) == len(v):
                try:
                    import math
                    dot = sum(a*b for a, b in zip(q, v))
                    nq = math.sqrt(sum(a*a for a in q))
                    nv = math.sqrt(sum(a*a for a in v))
                    if nq > 0 and nv > 0:
                        sim = dot/(nq*nv)
                        dist = 1.0 - sim
                    else:
                        dist = sum((a-b)**2 for a,b in zip(q, v))
                except _CHROMA_NONCRITICAL_EXCEPTIONS:
                    dist = 0.0
            items.append((dist, k))
        items.sort(key=lambda x: x[0])
        keys = [k for _, k in items[:n_results]]
        out = {"ids": [keys]}
        if "documents" in include:
            out["documents"] = [[self._docs[k] for k in keys]]
        if "metadatas" in include:
            out["metadatas"] = [[self._meta.get(k, {}) for k in keys]]
        if "embeddings" in include:
            out["embeddings"] = [[self._embs.get(k, []) for k in keys]]
        if "distances" in include:
            # Use computed distances for top-k keys
            dmap = {k: d for d, k in items}
            out["distances"] = [[dmap.get(k, 0.0) for k in keys]]
        return out


class _InMemoryChromaClient:
    def __init__(self):
        self._collections: dict[str, _InMemoryCollection] = {}
        self._system = _InMemorySystem()

    def close(self):
        return None

    def list_collections(self) -> list[_InMemoryCollection]:
        return list(self._collections.values())

    def get_or_create_collection(self, name: str, metadata: Optional[dict[str, Any]] = None) -> _InMemoryCollection:
        if name not in self._collections:
            self._collections[name] = _InMemoryCollection(name, metadata=metadata)
        else:
            if metadata:
                with contextlib.suppress(_CHROMA_NONCRITICAL_EXCEPTIONS):
                    self._collections[name].modify(metadata=metadata)
        return self._collections[name]

    def create_collection(self, name: str, metadata: Optional[dict[str, Any]] = None) -> _InMemoryCollection:
        if name in self._collections:
            return self._collections[name]
        col = _InMemoryCollection(name, metadata=metadata)
        self._collections[name] = col
        return col

    def get_collection(self, name: str) -> _InMemoryCollection:
        if name not in self._collections:
            raise KeyError(f"Collection {name} does not exist")
        return self._collections[name]

    def delete_collection(self, name: str) -> None:
        col = self._collections.pop(name, None)
        if col is not None:
            col._deleted = True
        return None

def get_default_chroma_manager():
    """Get or create the default ChromaDB manager for backward compatibility."""
    global _default_chroma_manager
    with _manager_lock:
        if _default_chroma_manager is None:
            # Use default user ID 1 for single-user mode
            from tldw_Server_API.app.core.config import settings
            user_id = str(settings.get("SINGLE_USER_FIXED_ID", "1"))
            # Get the embedding config and add USER_DB_BASE_DIR from main settings
            embedding_config = settings.get("EMBEDDING_CONFIG", {}).copy()
            embedding_config["USER_DB_BASE_DIR"] = settings.get("USER_DB_BASE_DIR")
            _default_chroma_manager = ChromaDBManager(user_id=user_id, user_embedding_config=embedding_config)
        return _default_chroma_manager

# Legacy function exports for backward compatibility
def store_in_chroma(texts, embeddings, ids, metadatas, collection_name="default_collection"):
    """Legacy function for storing embeddings in ChromaDB."""
    manager = get_default_chroma_manager()
    return manager.store_in_chroma(collection_name, texts, embeddings, ids, metadatas)

def get_default_chroma_client():
    """Return the default manager's underlying Chroma client instance."""
    manager = get_default_chroma_manager()
    return getattr(manager, "client", None)
