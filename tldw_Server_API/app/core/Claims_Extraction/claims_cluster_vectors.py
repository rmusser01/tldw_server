"""Strict existing-only vector reader; no owner provisioning or stub fallback."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from chromadb.errors import ChromaError, NotFoundError

from tldw_Server_API.app.core.DB_Management.chroma_read_validation import ChromaPreflightLimits
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import ClusterRebuildStorageError
from tldw_Server_API.app.core.Embeddings.chroma_storage import ChromaStorageError
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager


def _translate(exc: BaseException) -> ClusterRebuildStorageError:
    if isinstance(exc, ChromaStorageError):
        return ClusterRebuildStorageError(f"claims_cluster_{exc.code}", retryable=exc.retryable)
    return ClusterRebuildStorageError("claims_cluster_vector_access", retryable=True)


@contextmanager
def open_claim_cluster_vector_reader(
    *, owner_user_id: str, embedding_config: Mapping[str, Any], preflight_limits: ChromaPreflightLimits | None = None
) -> Iterator[Callable[[tuple[str, ...]], Mapping[str, Sequence[float]]] | None]:
    """Yield a bounded callback inside one coordinated read/SDK lifetime.

    Use on one synchronous thread (including the whole context when offloading).
    None means proven absent owner storage or collection, never a provider error.
    Native-lock metadata may be established beside, never inside, the owner tree.
    """
    base = embedding_config.get("USER_DB_BASE_DIR")
    if not base:
        raise ClusterRebuildStorageError("claims_cluster_vector_configuration", retryable=False)
    try:
        path = DatabasePaths.resolve_user_chroma_directory(owner_user_id, base_dir_override=base)
    except ValueError as exc:
        raise ClusterRebuildStorageError("claims_cluster_vector_configuration", retryable=False) from exc
    except OSError as exc:
        raise _translate(exc) from exc
    try:
        path.stat()
    except FileNotFoundError:
        yield None
        return
    except OSError as exc:
        raise _translate(exc) from exc
    manager = None
    try:
        manager = ChromaDBManager(
            owner_user_id, dict(embedding_config), existing_only=True, preflight_limits=preflight_limits
        )
        with manager.storage_access():
            try:
                collection = manager.get_existing_collection_strict(f"claims_for_{owner_user_id}")
            except NotFoundError:
                yield None
                return

            def read(ids: tuple[str, ...]) -> Mapping[str, tuple[float, ...]]:
                if len(ids) > 200 or any(not isinstance(key, str) or len(key) > 512 for key in ids):
                    raise ClusterRebuildStorageError("claims_cluster_resource_limit", retryable=False)
                if not ids:
                    return {}
                try:
                    result = collection.get(ids=ids, include=["embeddings"])
                except (ChromaStorageError, ChromaError, OSError, RuntimeError, ValueError) as exc:
                    raise _translate(exc) from exc
                returned_ids = result.get("ids")
                vectors = result.get("embeddings")
                if returned_ids is None or vectors is None or len(returned_ids) != len(vectors):
                    raise ClusterRebuildStorageError("claims_cluster_vector_corruption", retryable=False)
                if (
                    len(returned_ids) > len(ids)
                    or len(set(returned_ids)) != len(returned_ids)
                    or any(key not in ids for key in returned_ids)
                ):
                    raise ClusterRebuildStorageError("claims_cluster_vector_corruption", retryable=False)
                mapping = {}
                for key, vector in zip(returned_ids, vectors, strict=True):
                    if vector is None or len(vector) > 4096:
                        raise ClusterRebuildStorageError("claims_cluster_vector_corruption", retryable=False)
                    mapping[key] = tuple(vector)
                return mapping

            yield read
    except ClusterRebuildStorageError:
        raise
    except (ChromaStorageError, ChromaError, OSError, RuntimeError, ValueError) as exc:
        raise _translate(exc) from exc
    finally:
        if manager is not None:
            manager.close()
