# kanban_vector_search.py
# Description: Optional ChromaDB vector search integration for Kanban cards.
#
"""
kanban_vector_search.py
-----------------------

Provides optional vector search functionality for Kanban cards using ChromaDB.
Gracefully degrades to FTS-only search when ChromaDB is not available.

Collection name: kanban_user_{safe_user_id}
Document: Card title + description + label names
Metadata: card_id, board_id, list_id, due_date, priority, labels, created_at
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from typing import Any

from functools import lru_cache

from loguru import logger

from tldw_Server_API.app.core.Embeddings import redis_pipeline
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.testing import is_test_mode

# ChromaDB components are resolved on first use rather than at import time.
# Importing ChromaDB_Library here loaded chromadb (~0.25 s) in every process
# that imported Kanban_DB, which route registration does at startup.
@lru_cache(maxsize=1)
def _resolve_chromadb_manager() -> type | None:
    """Return the ChromaDBManager class, or None when ChromaDB is unavailable."""
    try:
        from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import (
            ChromaDBManager,
        )
        return ChromaDBManager
    except ImportError as e:
        logger.warning(f"ChromaDB not available for Kanban vector search: {e}")
    except Exception as e:
        logger.warning(f"ChromaDB initialization error for Kanban: {e}")
    return None


KANBAN_COLLECTION_PREFIX = "kanban_user_"
_EMBEDDINGS_DOMAIN = "embeddings"
_EMBEDDINGS_ROOT_JOB_TYPE = "embeddings_pipeline"


def _is_noncritical_vector_error(exc: BaseException) -> bool:
    """Return True for errors where vector-search should degrade gracefully."""
    if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        return False
    if isinstance(exc, Exception):
        return True
    # pyo3_runtime.PanicException subclasses BaseException.
    return exc.__class__.__name__ == "PanicException"


def is_vector_search_available() -> bool:
    """Check if vector search is available."""
    return _resolve_chromadb_manager() is not None


def get_kanban_collection_name(user_id: str) -> str:
    """
    Get the ChromaDB collection name for a user's Kanban cards.

    Sanitizes the user_id for use in collection names by replacing
    hyphens and spaces with underscores. User IDs longer than 50 characters
    are shortened with a hash suffix to reduce collision risk while
    respecting collection name limits.

    Args:
        user_id: The user identifier (can contain hyphens, spaces, etc.)

    Returns:
        A sanitized collection name in the format 'kanban_user_{safe_user_id}'.
    """
    user_id_str = str(user_id)
    sanitized = user_id_str.replace("-", "_").replace(" ", "_")
    if len(sanitized) > 50:
        hash_suffix = hashlib.sha256(user_id_str.encode("utf-8")).hexdigest()[:16]
        prefix_len = 50 - 1 - len(hash_suffix)
        safe_user_id = f"{sanitized[:prefix_len]}_{hash_suffix}"
    else:
        safe_user_id = sanitized
    return f"{KANBAN_COLLECTION_PREFIX}{safe_user_id}"


def _jobs_queue() -> str:
    """Return the configured embeddings stage queue name."""
    queue = (os.getenv("EMBEDDINGS_JOBS_QUEUE") or "default").strip()
    return queue or "default"


def _root_jobs_queue(stage_queue: str) -> str:
    """Select the root jobs queue, defaulting to low priority when stages aren't low."""
    root_queue = (os.getenv("EMBEDDINGS_ROOT_JOBS_QUEUE") or "").strip()
    if root_queue:
        return root_queue
    return "low" if stage_queue != "low" else "default"


def _jobs_manager() -> JobManager:
    """Create a JobManager using the configured Jobs DB when available."""
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    if not db_url:
        return JobManager()
    backend = "postgres" if db_url.startswith("postgres") else None
    return JobManager(backend=backend, db_url=db_url)


def _map_priority(priority: int) -> int:
    """Map a 0-100 style priority to the 1-10 scale used by the jobs system."""
    try:
        val = int(priority)
    except (TypeError, ValueError):
        val = 50
    mapped = max(1, min(10, int(val / 10)))
    return mapped or 5


def _extract_embedding_settings(embedding_config: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Pull provider/model identifiers from a user embedding configuration."""
    if not isinstance(embedding_config, dict):
        return None, None
    model = (
        embedding_config.get("embedding_model")
        or embedding_config.get("default_model_id")
        or embedding_config.get("model_id")
    )
    provider = embedding_config.get("embedding_provider") or embedding_config.get("provider")
    return model, provider


class KanbanVectorSearch:
    """
    Provides vector search functionality for Kanban cards.

    Wraps ChromaDBManager with Kanban-specific logic for:
    - Building searchable documents from cards
    - Filtering by board, labels, priority
    - Graceful degradation when unavailable
    """

    def __init__(
        self,
        user_id: str,
        embedding_config: dict[str, Any] | None = None,
    ):
        """
        Initialize the Kanban vector search.

        Args:
            user_id: The user ID for isolation.
            embedding_config: Configuration for ChromaDB/embeddings.
                             If None, vector search will be disabled.
        """
        self.user_id = str(user_id)
        self.embedding_config = embedding_config
        self._manager: Any | None = None
        self._collection_name = get_kanban_collection_name(self.user_id)
        self._available = False

        chromadb_manager_cls = _resolve_chromadb_manager()
        if chromadb_manager_cls is None:
            logger.debug(f"KanbanVectorSearch for user {self.user_id}: ChromaDB not available")
            return

        if not embedding_config:
            logger.debug(f"KanbanVectorSearch for user {self.user_id}: No embedding config provided")
            return

        try:
            self._manager = chromadb_manager_cls(
                user_id=self.user_id,
                user_embedding_config=embedding_config,
            )
            self._available = True
            logger.info(f"KanbanVectorSearch initialized for user {self.user_id}")
        except BaseException as e:
            if not _is_noncritical_vector_error(e):
                raise
            logger.warning(f"KanbanVectorSearch init failed for user {self.user_id}: {e}")

    @property
    def available(self) -> bool:
        """Check if vector search is available for this instance."""
        return self._available and self._manager is not None

    def close(self) -> None:
        """Close the ChromaDB manager."""
        if self._manager is not None:
            try:
                self._manager.close()
            except Exception as e:
                logger.warning(f"Error closing KanbanVectorSearch for user {self.user_id}: {e}")
            finally:
                self._manager = None
                self._available = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _build_document(self, card: dict[str, Any]) -> str:
        """
        Build a searchable document from card data.

        Combines title, description, label names, and checklist item names for embedding.
        """
        parts = []

        # Title is required
        title = card.get("title", "")
        if title:
            parts.append(title)

        # Description is optional
        description = card.get("description", "")
        if description:
            parts.append(description)

        # Add label names if present
        labels = card.get("labels", [])
        if labels:
            label_names = [label.get("name", "") for label in labels if label.get("name")]
            if label_names:
                parts.append("Labels: " + ", ".join(label_names))

        checklist_items = card.get("checklist_items", [])
        if checklist_items:
            item_names = [item for item in checklist_items if item]
            if item_names:
                parts.append("Checklist: " + "; ".join(item_names))

        return " ".join(parts)

    def _build_metadata(self, card: dict[str, Any]) -> dict[str, Any]:
        """
        Build ChromaDB metadata from card data.

        Only includes serializable values.
        """
        metadata = {
            "card_id": card.get("id"),
            "board_id": card.get("board_id"),
            "list_id": card.get("list_id"),
        }

        # Optional fields
        if card.get("priority"):
            metadata["priority"] = card["priority"]
        if card.get("due_date"):
            metadata["due_date"] = card["due_date"]
        if card.get("created_at"):
            metadata["created_at"] = card["created_at"]
        labels = card.get("labels", [])
        if labels:
            label_names = [label.get("name") for label in labels if label.get("name")]
            if label_names:
                metadata["labels"] = label_names

        return metadata

    def index_card(self, card: dict[str, Any]) -> bool:
        """
        Index a card for vector search.

        Args:
            card: Card data including id, title, description, labels, etc.

        Returns:
            True if indexed successfully, False otherwise.
        """
        if not self.available:
            return False

        try:
            card_id = card.get("id")
            if not card_id:
                logger.warning(f"KanbanVectorSearch.index_card: missing or invalid card id for user {self.user_id}")
                return False

            if is_test_mode():
                return True

            doc_id = f"card_{card_id}"
            document = self._build_document(card)
            metadata = self._build_metadata(card)
            embedding_model, embedding_provider = _extract_embedding_settings(self.embedding_config)

            payload = {
                "content": document,
                "metadata": metadata,
                "collection_name": self._collection_name,
                "document_id": doc_id,
                "card_id": card_id,
                "card_version": card.get("version"),
                "current_stage": "content",
                "request_source": "kanban",
            }
            if embedding_model:
                payload["embedding_model"] = embedding_model
            if embedding_provider:
                payload["embedding_provider"] = embedding_provider

            jm = _jobs_manager()
            stage_queue = _jobs_queue()
            root_queue = _root_jobs_queue(stage_queue)
            root_job = jm.create_job(
                domain=_EMBEDDINGS_DOMAIN,
                queue=root_queue,
                job_type=_EMBEDDINGS_ROOT_JOB_TYPE,
                payload=payload,
                owner_user_id=str(self.user_id),
                priority=_map_priority(50),
                max_retries=0,
            )
            root_uuid = str(root_job.get("uuid") or "")
            if not root_uuid:
                logger.warning(
                    f"KanbanVectorSearch.index_card: missing root job uuid for card {card_id} "
                    f"(user {self.user_id})"
                )
                return False
            stage_payload = dict(payload)
            stage_payload["root_job_uuid"] = root_uuid
            stage_payload["parent_job_uuid"] = root_uuid
            stage_payload["user_id"] = str(self.user_id)

            try:
                redis_pipeline.enqueue_content_job(
                    payload=stage_payload,
                    root_job_uuid=root_uuid,
                    force_regenerate=False,
                    require_redis=not redis_pipeline.allow_stub(),
                )
            except Exception:
                with contextlib.suppress(Exception):
                    jm.fail_job(
                        int(root_job["id"]),
                        error="Failed to enqueue kanban embeddings job to Redis",
                        retryable=False,
                        enforce=False,
                    )
                raise

            logger.debug(f"Queued embeddings for card {card_id} (user {self.user_id})")
            return True

        except Exception as e:
            logger.warning(f"Failed to index card {card.get('id')} for user {self.user_id}: {e}")
            return False

    def remove_card(self, card_id: int) -> bool:
        """
        Remove a card from the vector index.

        Args:
            card_id: The card ID to remove.

        Returns:
            True if removed successfully, False otherwise.
        """
        if not self.available:
            return False
        manager = self._manager
        if manager is None:
            return False

        try:
            doc_id = f"card_{card_id}"
            collection = manager.get_or_create_collection(self._collection_name)
            collection.delete(ids=[doc_id])

            logger.debug(f"Removed card {card_id} from index for user {self.user_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to remove card {card_id} from index for user {self.user_id}: {e}")
            return False

    def search(
        self,
        query: str,
        board_id: int | None = None,
        priority: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Search cards using vector similarity.

        The returned ``relevance_score`` is a normalized similarity value in the
        range ``0.0`` to ``1.0``, derived from the underlying L2 distance
        reported by ChromaDB (lower distance → higher relevance).

        Args:
            query: The search query.
            board_id: Optional board ID filter.
            priority: Optional priority filter.
            limit: Maximum number of results.

        Returns:
            List of search results with card_id, board_id, list_id and
            relevance_score in the range [0.0, 1.0].
        """
        if not self.available:
            return []
        manager = self._manager
        if manager is None:
            return []

        try:
            # Build where filter
            where_filter: dict[str, Any] | None = None
            if board_id or priority:
                where_filter = {}
                if board_id:
                    where_filter["board_id"] = board_id
                if priority:
                    where_filter["priority"] = priority

            # Use the manager's vector search
            results = manager.vector_search(
                query=query,
                collection_name=self._collection_name,
                k=limit,
                where_filter=where_filter,
                include_fields=["metadatas", "distances"],
            )

            # Convert results to our format
            search_results = []
            for result in results:
                metadata = result.get("metadata", {})
                distance = result.get("distance", 1.0)

                # Convert distance to similarity score (lower distance = higher similarity)
                # ChromaDB uses L2 distance by default
                relevance_score = 1.0 / (1.0 + distance) if distance is not None else 0.0

                search_results.append({
                    "card_id": metadata.get("card_id"),
                    "board_id": metadata.get("board_id"),
                    "list_id": metadata.get("list_id"),
                    "relevance_score": relevance_score,
                })

            return search_results

        except Exception as e:
            logger.warning(f"Vector search failed for user {self.user_id}: {e}")
            return []

    def reindex_all_cards(self, cards: list[dict[str, Any]]) -> tuple[int, int]:
        """
        Reindex all cards for a user (useful for rebuilding the index).

        Args:
            cards: List of card dictionaries.

        Returns:
            Tuple of (success_count, failure_count).
        """
        if not self.available:
            return 0, len(cards)

        success = 0
        failure = 0

        for card in cards:
            if self.index_card(card):
                success += 1
            else:
                failure += 1

        logger.info(f"Reindexed {success} cards for user {self.user_id} ({failure} failures)")
        return success, failure


def create_kanban_vector_search(
    user_id: str,
    embedding_config: dict[str, Any] | None = None,
) -> KanbanVectorSearch | None:
    """
    Factory function to create a KanbanVectorSearch instance.

    Returns None if vector search is not available or not configured.

    Args:
        user_id: The user ID.
        embedding_config: Embedding configuration (from app config).

    Returns:
        KanbanVectorSearch instance or None.
    """
    if not is_vector_search_available():
        return None

    if not embedding_config:
        return None

    try:
        search = KanbanVectorSearch(user_id, embedding_config)
        return search if search.available else None
    except BaseException as e:
        if not _is_noncritical_vector_error(e):
            raise
        logger.warning(f"Failed to create KanbanVectorSearch for user {user_id}: {e}")
        return None
