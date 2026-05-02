from __future__ import annotations

from collections.abc import Iterable

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.RAG.rag_service.cache_invalidation import (
    collect_cache_namespaces as _collect_cache_namespaces,
)
from tldw_Server_API.app.core.RAG.rag_service.cache_invalidation import (
    invalidate_rag_caches as _core_invalidate_rag_caches,
)

_RAG_CACHE_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


async def delete_media_vectors(
    current_user: User | None,
    *,
    media_id: int,
    namespaces: Iterable[str] | None = None,
) -> None:
    """Best-effort vector-store cleanup for a media item."""
    cache_namespaces = _collect_cache_namespaces(current_user, namespaces)
    if not cache_namespaces:
        cache_namespaces = {"0"}

    try:
        from tldw_Server_API.app.core.config import settings as _settings
        if not isinstance(_settings, dict):
            return
        rag_cfg = _settings.get("RAG", {}) or {}
        if not rag_cfg.get("vector_store_type"):
            return
    except (ImportError, *_RAG_CACHE_NONCRITICAL_EXCEPTIONS) as exc:
        logger.debug("Vector cleanup skipped (settings error): {}", exc)
        return

    try:
        from tldw_Server_API.app.core.RAG.rag_service.vector_stores.factory import (
            create_from_settings_for_user,
        )
    except ImportError as exc:
        logger.debug("Vector cleanup skipped (factory import error): {}", exc)
        return

    for namespace in cache_namespaces:
        try:
            user_id = str(namespace)
            adapter = create_from_settings_for_user(_settings, user_id)
            if adapter is None:
                continue
            if not getattr(adapter, "_initialized", False):
                await adapter.initialize()
            collection_name = f"user_{user_id}_media_embeddings"
            await adapter.delete_by_filter(collection_name, {"media_id": str(media_id)})
        except _RAG_CACHE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(
                "Vector cleanup skipped for user {} media {}: {}",
                namespace,
                media_id,
                exc,
            )


def invalidate_rag_caches(
    current_user: User | None,
    *,
    namespaces: Iterable[str] | None = None,
    media_id: int | None = None,
) -> None:
    """Compatibility wrapper around the core cache invalidation helper."""

    _core_invalidate_rag_caches(
        current_user,
        namespaces=namespaces,
        media_id=media_id,
    )
