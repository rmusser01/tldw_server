"""
ChromaDB implementation of the VectorStoreAdapter interface.

This module provides a ChromaDB-specific implementation that wraps
the existing ChromaDBManager functionality.
"""

import asyncio
from collections.abc import Sequence
from typing import Any, Literal, Optional, cast

from loguru import logger

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

# Import existing ChromaDB implementation
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
from tldw_Server_API.app.core.Metrics.traces import get_tracing_manager

from .base import VectorSearchResult, VectorStoreAdapter, VectorStoreConfig


class ChromaDBNotInitializedError(RuntimeError):
    """Raised when the ChromaDB adapter is used before initialization."""

    def __init__(self) -> None:
        super().__init__("ChromaDB adapter not initialized")


class ChromaDBVectorsNotFoundError(ValueError):
    """Raised when requested vector IDs are missing from a collection."""

    def __init__(self, missing: list[str]) -> None:
        super().__init__(f"Vector(s) not found: {missing}")
        self.missing = missing


def _raise_missing_vectors(missing: list[str]) -> None:
    raise ChromaDBVectorsNotFoundError(missing)


class ChromaDBAdapter(VectorStoreAdapter):
    """ChromaDB implementation of the vector store adapter."""

    def __init__(self, config: VectorStoreConfig):
        """
        Initialize ChromaDB adapter.

        Args:
            config: Vector store configuration
        """
        super().__init__(config)
        self.manager: Optional[ChromaDBManager] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _require_manager(self) -> ChromaDBManager:
        if self.manager is None:
            raise ChromaDBNotInitializedError()
        return self.manager

    async def initialize(self) -> None:
        """Initialize ChromaDB connection."""
        try:
            # Always use a fresh manager bound to current settings (avoids cross-test leakage)
            from tldw_Server_API.app.core.config import settings
            user_id = self.config.user_id
            embedding_config = self.config.connection_params.get("embedding_config", {}).copy()
            try:
                embedding_config["USER_DB_BASE_DIR"] = str(DatabasePaths.get_user_db_base_dir())
            except Exception:  # noqa: BLE001 - preserve adapter initialization fallback
                embedding_config["USER_DB_BASE_DIR"] = settings.get("USER_DB_BASE_DIR")
            self.manager = ChromaDBManager(
                user_id=user_id,
                user_embedding_config=embedding_config
            )

            self._initialized = True
            self._loop = asyncio.get_event_loop()
            logger.info(f"ChromaDB adapter initialized for user {self.config.user_id}")

        except Exception:  # noqa: BLE001 - surface initialization failures
            logger.error("Failed to initialize ChromaDB adapter")
            raise

    async def create_collection(self, collection_name: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """
        Create a new collection in ChromaDB.

        Args:
            collection_name: Name of the collection to create
            metadata: Optional metadata for the collection
        """
        if not self._initialized:
            await self.initialize()

        try:
            manager = self._require_manager()
            # Add embedding dimension to metadata
            if metadata is None:
                metadata = {}
            metadata["embedding_dimension"] = self.config.embedding_dim

            # ChromaDBManager's get_or_create_collection handles creation
            tm = get_tracing_manager()
            with tm.span("vectorstore.chromadb.create_collection", attributes={
                "vs.collection": collection_name,
                "vs.embed_dim": int(self.config.embedding_dim),
            }):
                collection = manager.get_or_create_collection(collection_name)

            # Update metadata if provided
            if metadata and hasattr(collection, 'modify'):
                collection.modify(metadata=metadata)

            logger.info(f"Created/accessed collection '{collection_name}'")

        except Exception:
            logger.error("Failed to create ChromaDB collection")
            raise

    async def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection from ChromaDB.

        Args:
            collection_name: Name of the collection to delete
        """
        if not self._initialized:
            await self.initialize()

        try:
            manager = self._require_manager()
            manager.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
        except Exception:
            logger.error("Failed to delete ChromaDB collection")
            raise

    async def list_collections(self) -> list[str]:
        """
        List all collections in ChromaDB.

        Returns:
            List of collection names
        """
        if not self._initialized:
            await self.initialize()

        try:
            manager = self._require_manager()
            collections = manager.client.list_collections()
            return [col.name for col in collections]
        except Exception:
            logger.error("Failed to list ChromaDB collections")
            raise

    async def upsert_vectors(
        self,
        collection_name: str,
        ids: list[str],
        vectors: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]]
    ) -> None:
        """
        Insert or update vectors in ChromaDB collection.

        Args:
            collection_name: Target collection name
            ids: Unique identifiers for each vector
            vectors: Embedding vectors
            documents: Original text documents
            metadatas: Metadata for each vector
        """
        if not self._initialized:
            await self.initialize()

        try:
            manager = self._require_manager()
            # Validate vectors
            self._validate_vectors(vectors)
            tm = get_tracing_manager()
            with tm.span("vectorstore.chromadb.upsert", attributes={
                "vs.collection": collection_name,
                "vs.count": int(len(vectors)),
                "vs.embed_dim": int(self.config.embedding_dim),
            }):
                # Store in ChromaDB using existing manager
                manager.store_in_chroma(
                    collection_name=collection_name,
                    texts=documents,
                    embeddings=vectors,
                    ids=ids,
                    metadatas=metadatas
                )
            logger.info(f"Upserted {len(vectors)} vectors to collection '{collection_name}'")
        except Exception:
            logger.error("Failed to upsert ChromaDB vectors")
            raise

    async def delete_vectors(self, collection_name: str, ids: list[str]) -> None:
        """
        Delete vectors from ChromaDB collection.

        Args:
            collection_name: Target collection name
            ids: IDs of vectors to delete
        """
        if not self._initialized:
            await self.initialize()

        try:
            manager = self._require_manager()
            # Use get-only to avoid creating missing collections implicitly
            collection = manager.client.get_collection(name=collection_name)
            # Verify the vectors exist before attempting deletion
            try:
                data = collection.get(ids=ids, include=[])
                existing_ids = set(data.get("ids") or []) if isinstance(data, dict) else set()
                missing = [vid for vid in ids if vid not in existing_ids]
                if missing:
                    _raise_missing_vectors(missing)
            except Exception:
                # Bubble up as not-found for endpoint to translate to 404/400
                raise
            collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors from collection '{collection_name}'")
        except Exception:
            logger.error("Failed to delete ChromaDB vectors")
            raise

    async def delete_by_filter(self, collection_name: str, filter: dict[str, Any]) -> int:
        """Delete vectors by metadata filter (best-effort) in ChromaDB.

        Returns 0 when count is not available.
        """
        if not self._initialized:
            await self.initialize()
        try:
            manager = self._require_manager()
            collection = manager.get_or_create_collection(collection_name)
            delete = getattr(collection, 'delete', None)
            if callable(delete):
                delete(where=filter)
        except Exception:  # noqa: BLE001 - best-effort delete
            logger.error("Failed to delete ChromaDB vectors by filter")
        else:
            return 0
        return 0

    # Adapter-specific helper: list vectors with pagination
    async def list_vectors_paginated(self, collection_name: str, limit: int, offset: int, filter: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if not self._initialized:
            await self.initialize()
        manager = self._require_manager()
        collection = manager.get_or_create_collection(collection_name)
        total = 0
        try:
            total = int(collection.count())
        except Exception:  # noqa: BLE001 - count best-effort
            total = 0
        # Best-effort where filter if supported
        try:
            data = collection.get(limit=limit, offset=offset, include=["documents", "metadatas"], where=filter)
        except Exception:  # noqa: BLE001 - fallback when where unsupported
            data = collection.get(limit=limit, offset=offset, include=["documents", "metadatas"])
        items = []
        data_dict = cast(dict[str, Any], data) if isinstance(data, dict) else {}
        if data_dict.get('ids'):
            for i, vid in enumerate(data_dict['ids']):
                items.append({
                    'id': vid,
                    'content': (data_dict.get('documents') or [""])[i] if data_dict.get('documents') else "",
                    'metadata': (data_dict.get('metadatas') or [{}])[i] if data_dict.get('metadatas') else {},
                })
        return {'items': items, 'total': total}

    # Adapter-specific helper: get single vector by id
    async def get_vector(self, collection_name: str, vector_id: str) -> Optional[dict[str, Any]]:
        if not self._initialized:
            await self.initialize()
        try:
            manager = self._require_manager()
            collection = manager.client.get_collection(name=collection_name)
            data = collection.get(ids=[vector_id], include=["documents", "metadatas"])
            data_dict = cast(dict[str, Any], data) if isinstance(data, dict) else {}
            ids = data_dict.get('ids')
            if not ids:
                return None
            content = (data_dict.get('documents') or [""])[0]
            metadata = (data_dict.get('metadatas') or [{}])[0]
        except Exception:  # noqa: BLE001 - best-effort lookup
            return None
        else:
            return {'id': vector_id, 'content': content, 'metadata': metadata}

    # Adapter-specific helper: for duplication - vectors plus embeddings
    async def list_vectors_with_embeddings_paginated(self, collection_name: str, limit: int, offset: int, filter: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if not self._initialized:
            await self.initialize()
        manager = self._require_manager()
        collection = manager.get_or_create_collection(collection_name)
        total = 0
        try:
            total = int(collection.count())
        except Exception:  # noqa: BLE001 - count best-effort
            total = 0
        try:
            data = collection.get(limit=limit, offset=offset, include=["embeddings", "documents", "metadatas"], where=filter)
        except Exception:  # noqa: BLE001 - fallback when where unsupported
            data = collection.get(limit=limit, offset=offset, include=["embeddings", "documents", "metadatas"])
        items = []
        data_dict = cast(dict[str, Any], data) if isinstance(data, dict) else {}
        if data_dict.get('ids'):
            embs = data_dict.get('embeddings') or []
            # Convert numpy arrays to lists if needed
            try:
                if hasattr(embs, 'tolist'):
                    embs = embs.tolist()
            except Exception:  # noqa: BLE001 - best-effort conversion
                logger.debug("Chroma adapter failed to convert embeddings to list")
            for i, vid in enumerate(data_dict['ids']):
                vec: list[float] = []
                try:
                    vec = list(embs[i]) if isinstance(embs[i], (list, tuple)) else (embs[i].tolist() if hasattr(embs[i], 'tolist') else [])
                except Exception:  # noqa: BLE001 - best-effort conversion
                    vec = []
                items.append({
                    'id': vid,
                    'vector': vec,
                    'content': (data_dict.get('documents') or [""])[i] if data_dict.get('documents') else "",
                    'metadata': (data_dict.get('metadatas') or [{}])[i] if data_dict.get('metadatas') else {},
                })
        return {'items': items, 'total': total}

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        k: int = 10,
        filter: Optional[dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> list[VectorSearchResult]:
        """
        Search for similar vectors in ChromaDB collection.

        Args:
            collection_name: Collection to search in
            query_vector: Query embedding vector
            k: Number of results to return
            filter: Optional metadata filters
            include_metadata: Whether to include metadata in results

        Returns:
            List of search results ordered by similarity
        """
        if not self._initialized:
            await self.initialize()

        try:
            manager = self._require_manager()
            collection = manager.get_or_create_collection(collection_name)

            # Prepare include fields
            IncludeField = Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]
            include_fields: list[IncludeField] = ["documents", "metadatas", "distances"]
            if not include_metadata:
                include_fields.remove("metadatas")

            # Perform search
            query_embeddings: list[Sequence[float]] = [query_vector]
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=k,
                where=filter,
                include=include_fields
            )

            # Convert to VectorSearchResult format
            search_results: list[VectorSearchResult] = []
            results_dict = cast(dict[str, Any], results) if isinstance(results, dict) else {}
            ids_block = results_dict.get('ids') or []
            if ids_block and ids_block[0]:
                for i in range(len(ids_block[0])):
                    result = VectorSearchResult(
                        id=ids_block[0][i],
                        content=(results_dict.get('documents') or [[""]])[0][i] if 'documents' in results_dict else "",
                        metadata=(results_dict.get('metadatas') or [[{}]])[0][i] if 'metadatas' in results_dict else {},
                        distance=(results_dict.get('distances') or [[0.0]])[0][i] if 'distances' in results_dict else 0.0,
                        score=1.0 - (results_dict.get('distances') or [[0.0]])[0][i] if 'distances' in results_dict else 1.0
                    )
                    search_results.append(result)

        except Exception:  # noqa: BLE001 - surface as adapter error
            logger.error("Failed to search ChromaDB collection")
            raise
        else:
            return search_results

    async def multi_search(
        self,
        collection_patterns: list[str],
        query_vector: list[float],
        k: int = 10,
        filter: Optional[dict[str, Any]] = None
    ) -> list[VectorSearchResult]:
        """
        Search across multiple collections matching patterns.

        Args:
            collection_patterns: Patterns to match collection names
            query_vector: Query embedding vector
            k: Number of results per collection
            filter: Optional metadata filters

        Returns:
            Merged list of search results from all matching collections
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Get all collections
            all_collections = await self.list_collections()

            # Filter collections by patterns
            matching_collections = []
            for pattern in collection_patterns:
                if pattern.endswith("*"):
                    prefix = pattern[:-1]
                    matching_collections.extend(
                        [col for col in all_collections if col.startswith(prefix)]
                    )
                else:
                    if pattern in all_collections:
                        matching_collections.append(pattern)

            # Remove duplicates
            matching_collections = list(set(matching_collections))

            if not matching_collections:
                logger.warning(f"No collections found matching patterns: {collection_patterns}")
                return []

            # Search in all matching collections
            all_results = []
            for collection_name in matching_collections:
                try:
                    results = await self.search(
                        collection_name=collection_name,
                        query_vector=query_vector,
                        k=k,
                        filter=filter,
                        include_metadata=True
                    )
                    # Add collection source to metadata
                    for result in results:
                        result.metadata["source_collection"] = collection_name
                    all_results.extend(results)
                except Exception:  # noqa: BLE001 - best-effort per-collection search
                    logger.warning("Failed to search ChromaDB collection during multi-search")
                    continue

            # Sort by score and return top k overall
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:k]

        except Exception:  # noqa: BLE001 - surface as adapter error
            logger.error("Failed to perform ChromaDB multi-search")
            raise

    async def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """
        Get statistics about a ChromaDB collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with stats (count, dimension, metadata)
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use get-only call; raise if collection does not exist
            manager = self._require_manager()
            collection = manager.client.get_collection(name=collection_name)

            # Get collection count
            count = collection.count()

            # Get collection metadata
            metadata = collection.metadata if hasattr(collection, 'metadata') else {}

            # Determine dimension: prefer metadata 'embedding_dimension', else sample, else config
            dimension = None
            try:
                if metadata and "embedding_dimension" in metadata:
                    dimension = int(metadata["embedding_dimension"]) or None
            except Exception:  # noqa: BLE001 - metadata best-effort
                dimension = None
            if dimension is None and count > 0:
                sample = collection.get(limit=1, include=["embeddings"])
                emb = sample.get("embeddings") if isinstance(sample, dict) else None
                emb_list: Optional[list[Any]] = None
                if isinstance(emb, list):
                    emb_list = emb
                elif emb is not None and hasattr(emb, "tolist"):
                    try:
                        emb_list = emb.tolist()
                    except Exception:  # noqa: BLE001 - best-effort conversion
                        emb_list = None
                if emb_list:
                    try:
                        dimension = len(emb_list[0])
                    except Exception:  # noqa: BLE001 - best-effort shape inspection
                        logger.debug("Chroma adapter failed to inspect embedding shape")
            if dimension is None:
                dimension = self.config.embedding_dim
        except Exception:  # noqa: BLE001 - surface as adapter error
            logger.error("Failed to get ChromaDB collection stats")
            raise
        else:
            return {
                "name": collection_name,
                "count": count,
                "dimension": dimension,
                "metadata": metadata,
                "distance_metric": self.config.distance_metric
            }

    async def optimize_collection(self, collection_name: str) -> None:
        """
        Optimize the collection for better search performance.
        ChromaDB handles optimization internally, so this is a no-op.

        Args:
            collection_name: Name of the collection to optimize
        """
        if not self._initialized:
            await self.initialize()

        # ChromaDB handles optimization automatically
        logger.info(f"ChromaDB auto-optimizes collection '{collection_name}'")

    async def get_index_info(self, collection_name: str) -> dict[str, Any]:
        # Chroma manages internal index structures; return generic info
        stats = await self.get_collection_stats(collection_name)
        return {
            'backend': 'chroma',
            'index_type': 'managed',
            'dimension': stats.get('dimension', self.config.embedding_dim),
            'count': stats.get('count', 0),
        }

    def set_ef_search(self, value: int) -> int:
        # No-op for Chroma
        return int(value)

    async def close(self) -> None:
        """Close ChromaDB connection."""
        # ChromaDB manager handles cleanup internally
        self._initialized = False
        logger.info("ChromaDB adapter closed")
