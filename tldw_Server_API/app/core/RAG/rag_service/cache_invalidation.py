from __future__ import annotations

from collections.abc import Iterable

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

_RAG_CACHE_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def collect_cache_namespaces(
    current_user: User | None,
    namespaces: Iterable[str] | None = None,
) -> set[str]:
    """Collect all namespaces that should be invalidated for a request."""

    collected: set[str] = set()
    for namespace in namespaces or []:
        namespace_text = str(namespace).strip()
        if namespace_text:
            collected.add(namespace_text)
    if current_user is None:
        return collected

    username = getattr(current_user, "username", None)
    if username is not None:
        username_text = str(username).strip()
        if username_text:
            collected.add(username_text)

    user_id = getattr(current_user, "id", None)
    if user_id is not None:
        user_id_text = str(user_id).strip()
        if user_id_text:
            collected.add(user_id_text)

    return collected


def invalidate_rag_caches(
    current_user: User | None,
    *,
    namespaces: Iterable[str] | None = None,
    media_id: int | None = None,
) -> None:
    """Best-effort RAG cache invalidation for content updates."""

    cache_namespaces = collect_cache_namespaces(current_user, namespaces)

    try:
        from tldw_Server_API.app.core.RAG.rag_service.semantic_cache import (
            clear_shared_caches,
        )

        if cache_namespaces:
            for namespace in sorted(cache_namespaces):
                clear_shared_caches(namespace=namespace)
        else:
            clear_shared_caches(namespace=None)
    except (ImportError, *_RAG_CACHE_NONCRITICAL_EXCEPTIONS) as exc:
        logger.debug("RAG cache invalidation skipped: {}", exc)

    if media_id is None:
        return

    try:
        from tldw_Server_API.app.core.RAG.rag_service.agentic_chunker import (
            invalidate_intra_doc_vectors,
        )

        invalidate_intra_doc_vectors(str(media_id))
    except (ImportError, *_RAG_CACHE_NONCRITICAL_EXCEPTIONS) as exc:
        logger.debug("Agentic cache invalidation skipped: {}", exc)
    else:
        logger.debug("RAG cache invalidation complete for media_id={}", media_id)
