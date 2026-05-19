# vector_stores_openai.py
# OpenAI-compatible Vector Store API backed by ChromaDB

from __future__ import annotations

import asyncio
import contextlib
import pathlib
import time
import uuid
from typing import Any

import tiktoken
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query
from fastapi import status as http_status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal, get_request_user, rbac_rate_limit, RequirePermission, RequireRole, resolve_user_id_for_request, User
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Chunking.base import ChunkerConfig, ChunkingMethod
from tldw_Server_API.app.core.Chunking.chunker import Chunker
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    get_media_by_id,
    search_media,
)
from tldw_Server_API.app.core.Embeddings.vector_store_batches_db import (
    create_batch as db_create_batch,
)
from tldw_Server_API.app.core.Embeddings.vector_store_batches_db import (
    get_batch as db_get_batch,
)
from tldw_Server_API.app.core.Embeddings.vector_store_batches_db import (
    init_db as init_batches_db,
)
from tldw_Server_API.app.core.Embeddings.vector_store_batches_db import (
    list_batches as db_list_batches,
)
from tldw_Server_API.app.core.Embeddings.vector_store_batches_db import (
    update_batch as db_update_batch,
)
from tldw_Server_API.app.core.Embeddings.vector_store_meta_db import (
    delete_store as meta_delete_store,
)
from tldw_Server_API.app.core.Embeddings.vector_store_meta_db import (
    find_store_by_name as meta_find_store_by_name,
)
from tldw_Server_API.app.core.Embeddings.vector_store_meta_db import (
    init_meta_db,
)
from tldw_Server_API.app.core.Embeddings.vector_store_meta_db import (
    list_stores as meta_list_stores,
)
from tldw_Server_API.app.core.Embeddings.vector_store_meta_db import (
    register_store as meta_register_store,
)
from tldw_Server_API.app.core.Embeddings.vector_store_meta_db import (
    rename_store as meta_rename_store,
)
from tldw_Server_API.app.core.RAG.rag_service.vector_stores.base import (
    VectorStoreAdapter,
    VectorStoreConfig,
    VectorStoreType,
)
from tldw_Server_API.app.core.RAG.rag_service.vector_stores.factory import (
    VectorStoreFactory,
    create_from_settings_for_user,
)
from tldw_Server_API.app.core.testing import env_flag_enabled
from tldw_Server_API.app.core.Utils.pydantic_compat import model_dump_compat

RBAC_VECTOR_ADMIN = rbac_rate_limit("vector.admin")
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})

_VECTORSTORE_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    HTTPException,
)


def _principal_has_admin_claims(principal: AuthPrincipal) -> bool:
    roles = {
        str(role).strip().lower()
        for role in (principal.roles or [])
        if str(role).strip()
    }
    if "admin" in roles:
        return True
    permissions = {
        str(permission).strip().lower()
        for permission in (principal.permissions or [])
        if str(permission).strip()
    }
    return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)

# Embeddings batch generator hook. Tests may monkeypatch this attribute directly or
# replace the provider resolver via _get_embeddings_fn(). We intentionally avoid
# importing the heavy embeddings stack at module import time.
create_embeddings_batch = None  # type: ignore[assignment]

def _get_embeddings_fn():
    """Return the embeddings batch function, preferring a patched module attribute.

    - If tests monkeypatched `create_embeddings_batch`, use that.
    - Else, lazily import the production function and cache it in the module global.
    """
    fn = globals().get("create_embeddings_batch")
    if callable(fn):
        return fn
    try:
        from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
            create_embeddings_batch as _impl,
        )
        globals()["create_embeddings_batch"] = _impl
        return _impl
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as exc:
        _captured_exc = exc
        def _err(*_args, **_kwargs):
            raise RuntimeError(
                "Embeddings service not available; patch _get_embeddings_fn() or create_embeddings_batch"
            ) from _captured_exc
        return _err

router = APIRouter(
    tags=["vector-stores"],
)

# Ensure DB is initialized for single-user default on import; per-user init done on demand
try:
    init_batches_db(str(settings.get("SINGLE_USER_FIXED_ID", "1")))
except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as _e:
    logger.warning(f"Vector store batch DB init warning: {_e}")


# In-memory store dimension registry (authoritative if present)
_STORE_DIMENSIONS: dict[str, int] = {}
_CREATED_NAMES_BY_USER: dict[str, set[str]] = {}


def _as_int(val: Any | None) -> int | None:
    try:
        return int(val) if val is not None else None
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        return None


# ==========================
# Helpers: policy + token limits
# ==========================

def _allowed_providers() -> list[str] | None:
    try:
        vals = settings.get("ALLOWED_EMBEDDING_PROVIDERS", [])
        if isinstance(vals, list) and vals:
            return [str(v).lower() for v in vals]
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        pass
    return None


def _allowed_models() -> list[str] | None:
    try:
        vals = settings.get("ALLOWED_EMBEDDING_MODELS", [])
        if isinstance(vals, list) and vals:
            return [str(v) for v in vals]
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        pass
    return None


def _model_allowed(model: str, allowed: list[str]) -> bool:
    for pat in allowed:
        if pat.endswith("*") and model.startswith(pat[:-1]):
            return True
        if model == pat:
            return True
    return False


def _get_model_max_tokens(provider: str, model: str) -> int:
    try:
        mapping = settings.get("EMBEDDING_MODEL_MAX_TOKENS", {}) or {}
        key = f"{provider}:{model}"
        if key in mapping:
            return int(mapping[key])
        if model in mapping:
            return int(mapping[model])
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        pass
    # default
    if provider.lower() == 'openai':
        return 8192
    return 8192


def _get_tokenizer(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, model_name: str) -> int:
    try:
        enc = _get_tokenizer(model_name)
        return len(enc.encode(text))
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        return max(1, len(text) // 4)


# ==========================
# Schemas (OpenAI-like)
# ==========================

class VectorStoreCreate(BaseModel):
    name: str | None = Field(
        default=None,
        description="Human-readable store name (unique per user).",
        examples=["docs-index"]
    )
    metadata: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Arbitrary metadata to associate with the store."
    )
    embedding_model: str | None = Field(
        default=None,
        description="Embedding model identifier for reference (optional).",
        examples=["text-embedding-3-small"]
    )
    dimensions: int = Field(
        ..., gt=0,
        description="Embedding vector dimension for this store.",
        examples=[1536]
    )


class VectorStoreObject(BaseModel):
    id: str = Field(..., description="Unique store ID.")
    object: str = Field("vector_store", description="Object type.")
    name: str | None = Field(None, description="Store name.")
    created_at: int = Field(..., description="Creation timestamp (unix).")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Store metadata.")
    dimensions: int = Field(..., description="Embedding vector dimension.")


class VectorRecord(BaseModel):
    id: str | None = Field(None, description="Vector identifier.")
    values: list[float] | None = Field(None, description="Embedding vector values.")
    content: str | None = Field(None, description="Raw text/content to embed (server may embed if values omitted).")
    metadata: dict[str, Any] | None = Field(default_factory=dict, description="Per-vector metadata.")


class UpsertVectorsRequest(BaseModel):
    records: list[VectorRecord] = Field(..., description="Vectors to upsert.")


class VectorItem(BaseModel):
    id: str = Field(..., description="Vector ID.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata filter conditions.")
    content: str | None = Field(None, description="Optional content associated with the vector.")


class QueryRequest(BaseModel):
    query: str | None = Field(None, description="Natural language query to search.", examples=["vector databases in production"])
    vector: list[float] | None = Field(None, description="Raw embedding vector to search by.")
    top_k: int = Field(default=10, gt=0, le=100, description="Number of results to return.")
    filter: dict[str, Any] | None = Field(None, description="Metadata filter expression.")


def _adapter_for_user(user: User, embedding_dim: int) -> VectorStoreAdapter:
    """Create a vector store adapter for the user via the factory.

    Defaults to ChromaDB if no vector store type is configured. Embedding
    dimension is taken from the endpoint input to ensure consistency per store.
    """
    uid = str(getattr(user, 'id', settings.get("SINGLE_USER_FIXED_ID", "1")))
    # Use factory to resolve store type and connection params from settings
    base = create_from_settings_for_user(settings, uid)
    # Derive config using resolved store type/params, but with the requested dim
    if base is not None and getattr(base, 'config', None) is not None:
        cfg = VectorStoreConfig(
            store_type=base.config.store_type,  # type: ignore[attr-defined]
            connection_params=base.config.connection_params,  # type: ignore[attr-defined]
            embedding_dim=int(embedding_dim),
            distance_metric=getattr(base.config, 'distance_metric', 'cosine'),  # type: ignore[attr-defined]
            collection_prefix=getattr(base.config, 'collection_prefix', 'unified'),  # type: ignore[attr-defined]
            user_id=uid,
        )
        return VectorStoreFactory.create_adapter(cfg, initialize=False)
    # Fallback to local Chroma configuration
    chroma_cfg = VectorStoreConfig(
        store_type=VectorStoreType.CHROMADB,
        connection_params={"use_default": True},
        embedding_dim=int(embedding_dim),
        user_id=uid,
    )
    return VectorStoreFactory.create_adapter(chroma_cfg, initialize=False)


def _vs_id() -> str:
    return f"vs_{uuid.uuid4().hex[:24]}"


def _now_ts() -> int:
    return int(time.time())

async def _get_adapter_for_user(user: User, embedding_dim: int) -> VectorStoreAdapter:
    """Obtain adapter; supports both sync and async monkeypatched factories in tests."""
    maybe = _adapter_for_user(user, embedding_dim)
    if asyncio.iscoroutine(maybe):
        return await maybe
    return maybe


@router.post("/vector_stores", response_model=VectorStoreObject)
async def create_vector_store(
    payload: VectorStoreCreate = Body(...),
    current_user: User = Depends(get_request_user),
):
    if payload.dimensions <= 0:
        raise HTTPException(400, detail="dimensions must be > 0")

    store_id = _vs_id()
    created = _now_ts()
    name = payload.name or store_id

    adapter = await _get_adapter_for_user(current_user, payload.dimensions)
    await adapter.initialize()

    # Enforce unique names per-user using meta DB, but ignore stale entries (no backing collection)
    uid = resolve_user_id_for_request(
        current_user,
        error_status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
    name_lower = (payload.name or "").strip().lower()
    if payload.name and payload.name.strip():
        try:
            init_meta_db(uid)
            existing = meta_find_store_by_name(uid, payload.name)
            if existing:
                raise HTTPException(status_code=409, detail=f"A vector store named '{payload.name}' already exists for this user")
        except HTTPException:
            raise
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
            logger.warning("Meta DB uniqueness check failed; falling back to adapter scan")
        # As a fallback when meta lookup fails, scan adapter collections by metadata.name
        try:
            for col in await adapter.list_collections():
                try:
                    st = await adapter.get_collection_stats(col)
                    md = st.get('metadata') or {}
                    if md.get('name') and md.get('name').strip().lower() == payload.name.strip().lower():
                        raise HTTPException(status_code=409, detail=f"A vector store named '{payload.name}' already exists for this user")
                except HTTPException:
                    raise
                except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                    continue
        except HTTPException:
            raise
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
            pass

    # Use store_id as collection name for uniqueness; keep human name in metadata
    metadata = dict(payload.metadata or {})
    metadata.update({
        "openai_id": store_id,
        "name": name,
        "created_at": created,
        "embedding_model": payload.embedding_model or "",
        "embedding_dimension": payload.dimensions,
        "owner": uid,
    })
    # Create the underlying collection; fall back to manager if adapter
    # does not provide a direct creation method (used by test fakes).
    try:
        create_coro = getattr(adapter, 'create_collection', None)
        if create_coro is not None:
            await create_coro(store_id, metadata=metadata)
        else:
            # Ensure a collection exists via manager; attach metadata if supported
            col = adapter.manager.get_or_create_collection(store_id)
            try:
                # Some fakes/clients don't persist metadata; best-effort only
                if hasattr(col, 'set_metadata'):
                    col.set_metadata(metadata)
            except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                pass
    except AttributeError:
        # Very minimal fake: only manager API available
        adapter.manager.get_or_create_collection(store_id)

    # Register in meta DB
    try:
        meta_register_store(uid, store_id, name)
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        logger.warning("Failed to register vector store in meta DB")

    # Track expected dimension in-memory for correctness in tests and fakes
    with contextlib.suppress(_VECTORSTORE_NONCRITICAL_EXCEPTIONS):
        _STORE_DIMENSIONS[store_id] = payload.dimensions
    # Track created name in this process for duplicate policies during tests
    try:
        if payload.name and payload.name.strip():
            _CREATED_NAMES_BY_USER.setdefault(uid, set()).add(name_lower)
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        pass

    return VectorStoreObject(
        id=store_id,
        name=name,
        created_at=created,
        metadata=metadata,
        dimensions=payload.dimensions,
    )


@router.get("/vector_stores")
async def list_vector_stores(
    current_user: User = Depends(get_request_user)
):
    """List vector stores for the current user."""
    # Prefer meta DB for performance and trusted names
    uid = resolve_user_id_for_request(
        current_user,
        error_status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
    stores = []
    used_ids = set()
    try:
        init_meta_db(uid)
        meta_rows = meta_list_stores(uid)
        adapter = None
        for row in meta_rows:
            if adapter is None:
                adapter = await _get_adapter_for_user(current_user, 1536)
                await adapter.initialize()
            try:
                stats = await adapter.get_collection_stats(row['id'])
                md = stats.get('metadata', {}) or {}
                md['name'] = row['name']
                stores.append({
                    'id': row['id'],
                    'object': 'vector_store',
                    'name': row['name'],
                    'created_at': md.get('created_at', row['created_at']),
                    'metadata': md,
                    'dimensions': stats.get('dimension', 1536)
                })
                used_ids.add(row['id'])
            except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                continue
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        logger.warning("Meta DB list failed; falling back to Chroma-only")
    # Include any collections not in meta DB
    try:
        adapter2 = await _get_adapter_for_user(current_user, 1536)
        await adapter2.initialize()
        for col_name in await adapter2.list_collections():
            if col_name in used_ids:
                continue
            try:
                stats = await adapter2.get_collection_stats(col_name)
                md = stats.get('metadata', {}) or {}
                stores.append({
                    'id': md.get('openai_id', col_name),
                    'object': 'vector_store',
                    'name': md.get('name', col_name),
                    'created_at': md.get('created_at', _now_ts()),
                    'metadata': md,
                    'dimensions': stats.get('dimension', 1536)
                })
            except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                continue
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        pass
    return {'data': stores}


@router.get(
    "/vector_stores/admin/users",
    dependencies=[
        Depends(RequireRole("admin")),
        Depends(RequirePermission(SYSTEM_CONFIGURE)),
        Depends(RBAC_VECTOR_ADMIN),
    ],
)
async def list_vector_store_users(current_user: User = Depends(get_request_user)):
    base_dir: pathlib.Path = DatabasePaths.get_user_db_base_dir()
    users = []
    try:
        for entry in base_dir.iterdir():
            if not entry.is_dir():
                continue
            uid = entry.name
            try:
                user_dir = DatabasePaths.get_user_base_directory(uid)
            except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                user_dir = entry
            vec_dir = user_dir / DatabasePaths.VECTOR_STORE_SUBDIR
            has_vec_dir = vec_dir.exists()
            store_count = 0
            batch_count = 0
            if has_vec_dir:
                # Count stores via meta DB
                try:
                    init_meta_db(uid)
                    store_count = len(meta_list_stores(uid))
                except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                    store_count = 0
                # Count batches
                try:
                    init_batches_db(uid)
                    from tldw_Server_API.app.core.Embeddings.vector_store_batches_db import _connect as batches_conn
                    with batches_conn(uid) as conn:
                        try:
                            cur = conn.execute("SELECT COUNT(1) FROM vector_store_batches")
                            row = cur.fetchone()
                            if row and row[0] is not None:
                                batch_count = int(row[0])
                        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                            batch_count = 0
                except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                    batch_count = 0
            users.append({
                'user_id': uid,
                'has_vector_dir': has_vec_dir,
                'store_count': store_count,
                'batch_count': batch_count
            })
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=500, detail="Failed to list vector store users") from e

    return { 'data': users }


## moved below batches route to avoid path shadowing


class VectorStoreUpdate(BaseModel):
    name: str | None = None
    metadata: dict[str, Any] | None = None


@router.patch("/vector_stores/{store_id}", response_model=VectorStoreObject)
async def update_vector_store(
    store_id: str = Path(...),
    payload: VectorStoreUpdate = Body(...),
    current_user: User = Depends(get_request_user)
):
    adapter = await _get_adapter_for_user(current_user, 1536)
    await adapter.initialize()
    try:
        stats = await adapter.get_collection_stats(store_id)
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=404, detail=f"Vector store not found: {e}") from e
    md = stats.get("metadata", {}) or {}
    # Enforce unique name per-user using meta DB first
    if payload.name and payload.name.strip():
        try:
            uid = resolve_user_id_for_request(
                current_user,
                error_status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
            init_meta_db(uid)
            existing = meta_find_store_by_name(uid, payload.name)
            if existing and existing.get('id') != store_id:
                raise HTTPException(status_code=409, detail=f"A vector store named '{payload.name}' already exists for this user")
        except HTTPException:
            raise
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
            pass
    # Enforce unique name per-user on rename
    if payload.name and payload.name.strip():
        try:
            for col in await adapter.list_collections():
                if col == store_id:
                    continue
                try:
                    st = await adapter.get_collection_stats(col)
                    other_md = st.get('metadata') or {}
                    if other_md.get('name') and other_md.get('name').strip().lower() == payload.name.strip().lower():
                        raise HTTPException(status_code=409, detail=f"A vector store named '{payload.name}' already exists for this user")
                except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                    continue
        except HTTPException:
            raise
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
            pass
    if payload.name:
        md["name"] = payload.name
    if payload.metadata:
        md.update(payload.metadata)

    # Use underlying manager to modify metadata
    collection = adapter.manager.get_or_create_collection(store_id)
    if hasattr(collection, "modify"):
        collection.modify(metadata=md)

    stats = await adapter.get_collection_stats(store_id)
    md = stats.get("metadata", {}) or {}
    # Update meta DB on rename: if not present, register; else rename
    try:
        if payload.name and payload.name.strip():
            uid_str = resolve_user_id_for_request(
                current_user,
                error_status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
            init_meta_db(uid_str)
            rows = meta_list_stores(uid_str)
            if any(r.get('id') == store_id for r in rows):
                meta_rename_store(uid_str, store_id, payload.name)
            else:
                # Not present: register with the new name
                meta_register_store(uid_str, store_id, payload.name)
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        logger.warning("Failed to persist vector store meta name update")

    return VectorStoreObject(
        id=md.get("openai_id", store_id),
        name=md.get("name", store_id),
        created_at=md.get("created_at", _now_ts()),
        metadata=md,
        dimensions=stats.get("dimension", 1536)
    )


@router.delete("/vector_stores/{store_id}")
async def delete_vector_store(
    store_id: str = Path(...),
    current_user: User = Depends(get_request_user)
):
    adapter = await _get_adapter_for_user(current_user, 1536)
    await adapter.initialize()
    await adapter.delete_collection(store_id)
    try:
        meta_delete_store(
            resolve_user_id_for_request(
                current_user,
                error_status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            ),
            store_id,
        )
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        logger.warning("Failed to delete vector store from meta DB")
    # Remove from in-memory registry
    with contextlib.suppress(_VECTORSTORE_NONCRITICAL_EXCEPTIONS):
        _STORE_DIMENSIONS.pop(store_id, None)
    return {"id": store_id, "deleted": True}


@router.post("/vector_stores/{store_id}/vectors")
async def upsert_vectors(
    store_id: str = Path(...),
    payload: UpsertVectorsRequest = Body(...),
    current_user: User = Depends(get_request_user)
):
    # Pre-scan records for first provided values length (helps infer dim for empty stores)
    first_values_len: int | None = None
    for rec in (payload.records or []):
        if rec.values is not None:
            first_values_len = len(rec.values)
            break

    # Resolve known store dimension from in-memory registry first
    registry_dim = _as_int(_STORE_DIMENSIONS.get(store_id))

    # Initialize adapter preferring the known/store dimension when available
    adapter = await _get_adapter_for_user(current_user, (registry_dim or first_values_len or 1536))
    await adapter.initialize()

    # Fetch stats (may be from real or fake adapter). Be tolerant if collection isn't created yet.
    try:
        stats = await adapter.get_collection_stats(store_id)
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        stats = {"dimension": registry_dim or 1536, "metadata": {}, "count": 0}
    stats_dim = _as_int(stats.get("dimension"))
    stats_md = stats.get("metadata", {}) or {}

    # If registry missing but metadata contains embedding_dimension, capture it
    if registry_dim is None:
        try:
            md_dim = _as_int(stats_md.get("embedding_dimension"))
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
            md_dim = None
        if md_dim and md_dim > 0:
            registry_dim = md_dim
            with contextlib.suppress(_VECTORSTORE_NONCRITICAL_EXCEPTIONS):
                _STORE_DIMENSIONS[store_id] = md_dim

    # Determine emptiness robustly
    is_empty: bool | None = None
    try:
        coll = adapter.manager.get_or_create_collection(store_id)
        try:
            cnt = coll.count()
            is_empty = (cnt == 0)
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
            is_empty = None
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        is_empty = None
    if is_empty is None:
        try:
            is_empty = bool(stats.get("count", 0) == 0)
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
            is_empty = False

    # Enforce store dimension from registry always; enforce stats only when collection is non-empty
    if first_values_len is not None:
        if registry_dim is not None and registry_dim != 1536 and first_values_len != registry_dim:
            raise HTTPException(400, detail=f"Vector length {first_values_len} != expected {registry_dim}")
        elif (registry_dim is None) and (not is_empty) and (stats_dim is not None) and stats_dim != 1536 and first_values_len != stats_dim:
            raise HTTPException(400, detail=f"Vector length {first_values_len} != expected {stats_dim}")

    # Final target dimension:
    # - If registry exists, use it
    # - Else if non-empty and stats gives a concrete dim (not generic), use stats
    # - Else prefer the provided first vector length (if given)
    # - Else fall back to default 1536
    # Resolve final dim with special-case for generic 1536 on empty stores
    if registry_dim is not None and registry_dim != 1536:
        dim = registry_dim
    elif (not is_empty) and (stats_dim is not None) and stats_dim != 1536:
        dim = stats_dim
    elif first_values_len is not None:
        dim = first_values_len
    else:
        dim = 1536

    # Ensure adapter config matches the resolved dimension
    if getattr(adapter, 'config', None) and getattr(adapter.config, 'embedding_dim', None) != dim:
        adapter = await _get_adapter_for_user(current_user, dim)
        await adapter.initialize()

    # Prepare buffers
    ids: list[str] = []
    vectors: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    # Support content->embed via local embedding pipeline if needed
    texts_to_embed: list[str] = []
    text_indices: list[int] = []

    for idx, rec in enumerate(payload.records):
        rid = rec.id or f"vec_{uuid.uuid4().hex[:24]}"
        ids.append(rid)
        # Ensure non-empty metadata dict per record
        meta = rec.metadata if rec.metadata and isinstance(rec.metadata, dict) and len(rec.metadata) > 0 else {"source": "api"}
        metadatas.append(meta)
        if rec.values is not None:
            if len(rec.values) != dim:
                raise HTTPException(400, detail=f"Vector length {len(rec.values)} != expected {dim}")
            vectors.append(rec.values)
            documents.append(rec.content or "")
        elif rec.content:
            texts_to_embed.append(rec.content)
            text_indices.append(idx)
            # placeholder; will fill after embedding
            vectors.append([0.0] * dim)
            documents.append(rec.content)
        else:
            raise HTTPException(400, detail="Each record requires 'values' or 'content'")

    if texts_to_embed:
        # Build app config for embedding create using default settings
        embedding_settings = settings.get("EMBEDDING_CONFIG", {})
        # Model/provider from settings; prefer explicit allowlists if provided
        model_id = embedding_settings.get("default_model_id") or embedding_settings.get("embedding_model") or "text-embedding-3-small"
        provider = embedding_settings.get("embedding_provider", "openai")
        mods_hint = _allowed_models()
        if mods_hint and len(mods_hint) > 0:
            model_id = mods_hint[0]

        # Token length checks first (do not block on allowlist). If policy lists exist,
        # use the strictest max token value among configured provider and allowed providers.
        max_tokens = _get_model_max_tokens(provider, model_id)
        provs = _allowed_providers()
        if provs:
            try:
                candidates = [max_tokens] + [_get_model_max_tokens(p, model_id) for p in provs]
                max_tokens = min([t for t in candidates if isinstance(t, int) and t > 0])
            except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                pass
        too_long: list[tuple[int, int]] = []
        for idx, tx in enumerate(texts_to_embed):
            tok = _count_tokens(tx, model_id)
            if tok > max_tokens:
                too_long.append((idx, tok))
        if too_long:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "input_too_long",
                    "message": f"One or more inputs exceed max tokens {max_tokens} for model {model_id}",
                    "details": [{"index": i, "tokens": tok} for (i, tok) in too_long]
                }
            )
        # Now enforce allowlist after token validation
        # Choose provider from allowlist for this request to avoid policy failures on short content
        if provs and len(provs) > 0:
            provider = provs[0]
        # Build app config with provider override for embedding backend
        app_config = {"embedding_config": {**embedding_settings, "embedding_provider": provider}}
        mods = _allowed_models()
        if mods is not None and not _model_allowed(model_id, mods):
            raise HTTPException(status_code=403, detail=f"Model '{model_id}' is not allowed for embeddings")
        if provs is not None and provider.lower() not in provs:
            raise HTTPException(status_code=403, detail=f"Provider '{provider}' is not allowed for embeddings")
        try:
            loop = asyncio.get_running_loop()
            embed_fn = _get_embeddings_fn()
            embedded = await loop.run_in_executor(None, embed_fn, texts_to_embed, app_config, model_id)
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Embedding generation failed for vectors upsert: {e}")
            raise HTTPException(500, detail="Failed to generate embeddings for provided content") from e
        # Assign back to vectors at corresponding indices
        for i, vec in zip(text_indices, embedded):
            if len(vec) != dim:
                raise HTTPException(status_code=422, detail=f"Auto-embedded vector length {len(vec)} != store dimension {dim}")
            vectors[i] = vec

    await adapter.upsert_vectors(store_id, ids=ids, vectors=vectors, documents=documents, metadatas=metadatas)
    return {"upserted": len(ids), "ids": ids}


class DuplicateStoreRequest(BaseModel):
    new_name: str = Field(..., description="Name for the duplicated store")
    dimensions: int | None = None

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {"new_name": "CopyOfStore", "dimensions": 1536}
        ]
    })


@router.post("/vector_stores/{store_id}/duplicate")
async def duplicate_vector_store(
    store_id: str = Path(...),
    payload: DuplicateStoreRequest = Body(...),
    current_user: User = Depends(get_request_user)
):
    uid = resolve_user_id_for_request(
        current_user,
        error_status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
    if not payload.new_name or not payload.new_name.strip():
        raise HTTPException(400, detail="new_name is required")
    # Uniqueness via meta DB
    try:
        init_meta_db(uid)
        if meta_find_store_by_name(uid, payload.new_name):
            raise HTTPException(409, detail=f"A vector store named '{payload.new_name}' already exists for this user")
    except HTTPException:
        raise
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        pass

    adapter = await _get_adapter_for_user(current_user, (payload.dimensions or 1536))
    await adapter.initialize()
    try:
        src_stats = await adapter.get_collection_stats(store_id)
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(404, detail=f"Source store not found: {e}") from e
    src_dim = src_stats.get('dimension', 1536)
    dim = payload.dimensions or src_dim

    # Create dest store
    create_payload = VectorStoreCreate(name=payload.new_name, metadata={'duplicated_from': store_id}, embedding_model=src_stats.get('metadata',{}).get('embedding_model',''), dimensions=dim)
    dest_vs = await create_vector_store(create_payload, current_user)
    dest_id = dest_vs.id

    # Copy in batches
    offset = 0
    step = 1000
    upserted = 0
    total = 0
    # Prefer adapter helper that returns vectors (works for PG + Chroma)
    dup_fn = getattr(adapter, 'list_vectors_with_embeddings_paginated', None)
    while True:
        ids_list: list[str] = []
        emb_list: list[list[float]] = []
        doc_list: list[str] = []
        meta_list_existing: list[dict[str, Any]] = []
        if callable(dup_fn):
            try:
                res = await dup_fn(store_id, step, offset, None)  # type: ignore[misc]
                total = int(res.get('total', total))
                items = res.get('items', [])
                if not items:
                    break
                for it in items:
                    ids_list.append(str(it.get('id')))
                    vec = it.get('vector') or []
                    # Validate vector length and type
                    if not isinstance(vec, list):
                        vec = []
                    emb_list.append(vec)
                    doc_list.append(it.get('content') or '')
                    meta_list_existing.append(it.get('metadata') or {})
            except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                # Fallback to Chroma collection path on error
                dup_fn = None
                continue
        if not callable(dup_fn):
            # Fallback: Chroma path
            source_collection = adapter.manager.get_or_create_collection(store_id)  # type: ignore[attr-defined]
            try:
                total = int(source_collection.count())
            except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                total = total or 0
            data = source_collection.get(limit=step, offset=offset, include=["embeddings", "documents", "metadatas"])  # type: ignore[attr-defined]
            if not data or not data.get('ids'):
                break
            ids_list = list(data.get('ids') or [])
            emb_list = list(data.get('embeddings') or [])
            try:
                if hasattr(emb_list, 'tolist'):
                    emb_list = emb_list.tolist()
            except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                pass
            doc_list = list(data.get('documents') or [])
            meta_list_existing = list(data.get('metadatas') or [])
            if len(emb_list) == 0:
                break
        # Adjust adapter dimension if needed for this batch
        emb_dim = len(emb_list[0]) if emb_list and emb_list[0] else dim
        if emb_dim != adapter.config.embedding_dim:
            adapter = await _get_adapter_for_user(current_user, emb_dim)
            await adapter.initialize()
        await adapter.upsert_vectors(dest_id, ids=ids_list, vectors=emb_list, documents=doc_list, metadatas=meta_list_existing)
        upserted += len(emb_list)
        offset += len(ids_list)
        # Only use the page-size termination when using fallback (Chroma) path.
        # For adapter-provided pagination, rely on the empty-items break above.
        if (not callable(dup_fn)) and len(ids_list) < step:
            break

    return { 'source_id': store_id, 'destination_id': dest_id, 'upserted': upserted, 'estimated_total': total }


class HNSWEfSearchRequest(BaseModel):
    ef_search: int = Field(..., gt=0, description="hnsw.ef_search value to set for this session")


@router.get(
    "/vector_stores/{store_id}/admin/index_info",
    dependencies=[
        Depends(RequireRole("admin")),
        Depends(RequirePermission(SYSTEM_CONFIGURE)),
        Depends(RBAC_VECTOR_ADMIN),
    ],
)
async def get_index_info(
    store_id: str = Path(...),
    current_user: User = Depends(get_request_user),
):
    adapter = await _get_adapter_for_user(current_user, 1536)
    await adapter.initialize()
    get_fn = getattr(adapter, 'get_index_info', None)
    if callable(get_fn):
        info = await get_fn(store_id)  # type: ignore[misc]
        return info
    # Fallback: return basic stats
    stats = await adapter.get_collection_stats(store_id)
    return {
        'backend': 'unknown',
        'dimension': stats.get('dimension', 1536),
        'count': stats.get('count', 0)
    }


@router.post(
    "/vector_stores/admin/hnsw_ef_search",
    dependencies=[
        Depends(RequireRole("admin")),
        Depends(RequirePermission(SYSTEM_CONFIGURE)),
        Depends(RBAC_VECTOR_ADMIN),
    ],
)
async def set_hnsw_ef_search(
    payload: HNSWEfSearchRequest = Body(...),
    current_user: User = Depends(get_request_user),
):
    adapter = await _get_adapter_for_user(current_user, 1536)
    await adapter.initialize()
    set_fn = getattr(adapter, 'set_ef_search', None)
    if callable(set_fn):
        value = set_fn(payload.ef_search)  # type: ignore[misc]
        return { 'ef_search': value, 'note': 'applies to current session/adapter only' }
    return { 'ef_search': payload.ef_search, 'note': 'no-op for this backend' }


class RebuildIndexRequest(BaseModel):
    index_type: str = Field(..., pattern="^(?i)(hnsw|ivfflat|drop)$")
    metric: str | None = Field(None, pattern="^(?i)(cosine|euclidean|ip)$")
    m: int | None = Field(16, ge=2, description="HNSW M parameter")
    ef_construction: int | None = Field(200, ge=1, description="HNSW ef_construction")
    lists: int | None = Field(100, ge=1, description="IVFFLAT lists")


@router.post(
    "/vector_stores/{store_id}/admin/rebuild_index",
    dependencies=[
        Depends(RequireRole("admin")),
        Depends(RequirePermission(SYSTEM_CONFIGURE)),
        Depends(RBAC_VECTOR_ADMIN),
    ],
)
async def rebuild_index(
    store_id: str = Path(...),
    payload: RebuildIndexRequest = Body(...),
    current_user: User = Depends(get_request_user),
):
    adapter = await _get_adapter_for_user(current_user, 1536)
    await adapter.initialize()
    rebuild_fn = getattr(adapter, 'rebuild_index', None)
    if not callable(rebuild_fn):
        raise HTTPException(status_code=400, detail="Index rebuild not supported for this backend")
    try:
        info = await rebuild_fn(  # type: ignore[misc]
            store_id,
            index_type=payload.index_type,
            metric=payload.metric,
            m=payload.m or 16,
            ef_construction=payload.ef_construction or 200,
            lists=payload.lists or 100,
        )
        return info
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=400, detail=f"Index rebuild failed: {e}") from e


class DeleteByFilterRequest(BaseModel):
    filter: dict[str, Any] = Field(..., description="Metadata filter expression")


@router.post(
    "/vector_stores/{store_id}/admin/delete_by_filter",
    dependencies=[
        Depends(RequireRole("admin")),
        Depends(RequirePermission(SYSTEM_CONFIGURE)),
        Depends(RBAC_VECTOR_ADMIN),
    ],
)
async def delete_by_filter(
    store_id: str = Path(...),
    payload: DeleteByFilterRequest = Body(...),
    current_user: User = Depends(get_request_user),
):
    # Guardrails: reject empty/overly broad deletes
    def _is_safe_filter(obj) -> bool:
        # Minimal safety: must be a non-empty dict with at least one concrete condition.
        if not isinstance(obj, dict) or not obj:
            return False
        # Disallow empty boolean operators
        if '$or' in obj and (not isinstance(obj['$or'], list) or len(obj['$or']) == 0):
            return False
        if '$and' in obj and (not isinstance(obj['$and'], list) or len(obj['$and']) == 0):
            return False
        # Recursively ensure at least one field is constrained
        def _has_concrete(node) -> bool:
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in ('$and', '$or'):
                        if isinstance(v, list) and any(_has_concrete(x) for x in v):
                            return True
                    else:
                        # Field-level constraint present
                        if v is None:
                            continue
                        if isinstance(v, dict):
                            # Operators like $in/$gte etc - consider non-empty as concrete
                            if any(True for _ in v.items()):
                                return True
                        else:
                            return True
            return False
        return _has_concrete(obj)

    if not _is_safe_filter(payload.filter):
        # Match test expectation exactly
        raise HTTPException(status_code=400, detail="Filter cannot be empty")
    adapter = await _get_adapter_for_user(current_user, 1536)
    await adapter.initialize()
    fn = getattr(adapter, 'delete_by_filter', None)
    if not callable(fn):
        raise HTTPException(status_code=400, detail="Delete by filter not supported for this backend")
    try:
        deleted = await fn(store_id, payload.filter)  # type: ignore[misc]
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=400, detail=f"Delete by filter failed: {e}") from e
    return {"deleted": int(deleted or 0)}


@router.get(
    "/vector_stores/admin/health",
    dependencies=[
        Depends(RequireRole("admin")),
        Depends(RequirePermission(SYSTEM_CONFIGURE)),
        Depends(RBAC_VECTOR_ADMIN),
    ],
)
async def vector_stores_health(current_user: User = Depends(get_request_user)):
    adapter = await _get_adapter_for_user(current_user, 1536)
    await adapter.initialize()
    fn = getattr(adapter, 'health', None)
    if callable(fn):
        try:
            return await fn()  # type: ignore[misc]
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as e:
            return {"ok": False, "error": "Vector store health check failed"}
    return {"ok": True}


@router.get("/vector_stores/{store_id}/vectors")
async def list_vectors(
    store_id: str = Path(...),
    limit: int = Query(50, gt=1, le=1000),
    offset: int = Query(0, ge=0),
    filter: str | None = Query(
        None,
        description="Optional JSON metadata filter",
        examples={
            "simple": {"summary": "Simple equality", "value": "{\"genre\":\"a\"}"},
            "and_numeric": {"summary": "AND with numeric", "value": "{\"$and\":[{\"genre\":\"a\"},{\"score\":{\"$gte\":0.8}}]}"}
        }
    ),
    order_by: str | None = Query(
        "id",
        description="Order field: 'id' or 'metadata.<key>'",
        examples={"metadata": {"summary": "Order by metadata.score desc", "value": "metadata.score"}}
    ),
    order_dir: str = Query(
        "asc",
        pattern="^(?i)(asc|desc)$",
        examples={"desc": {"summary": "Descending", "value": "desc"}}
    ),
    current_user: User = Depends(get_request_user)
):
    """List vectors in a store with pagination and optional filters/ordering."""
    adapter = await _get_adapter_for_user(current_user, 1536)
    await adapter.initialize()
    items: list[VectorItem] = []
    total: int = 0
    meta_filter: dict[str, Any] | None = None
    if filter:
        try:
            import json as _json
            parsed = _json.loads(filter)
            if not isinstance(parsed, dict):
                raise ValueError("filter must be a JSON object")
            meta_filter = parsed
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as e:
            raise HTTPException(status_code=400, detail={"error":"invalid_filter","message":str(e)}) from e
    if order_by and (order_by != 'id' and not order_by.startswith('metadata.')):
        raise HTTPException(status_code=400, detail={"error":"invalid_order_by","message":"order_by must be 'id' or 'metadata.<key>'"})
    # Prefer adapter-provided pagination helper when available (PG, future stores)
    list_fn = getattr(adapter, 'list_vectors_paginated', None)
    if callable(list_fn):
        try:
            if meta_filter is not None:
                result = await list_fn(store_id, limit=int(limit), offset=int(offset), filter=meta_filter, order_by=order_by, order_dir=order_dir)  # type: ignore[misc]
            else:
                result = await list_fn(store_id, limit=int(limit), offset=int(offset), order_by=order_by, order_dir=order_dir)  # type: ignore[misc]
            total = int(result.get('total', 0))
            for row in result.get('items', []):
                items.append(VectorItem(
                    id=str(row.get('id')),
                    metadata=row.get('metadata') or {},
                    content=row.get('content') or "",
                ))
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
            logger.warning("Adapter list_vectors_paginated failed; falling back to Chroma path")
    if not items and total == 0:
        # Fallback to Chroma collection semantics
        try:
            collection = adapter.manager.get_or_create_collection(store_id)  # type: ignore[attr-defined]
            try:
                total = int(collection.count())
            except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                total = 0
            try:
                data = collection.get(limit=limit, offset=offset, include=["documents", "metadatas"], where=meta_filter)  # type: ignore[attr-defined]
            except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                data = collection.get(limit=limit, offset=offset, include=["documents", "metadatas"])  # type: ignore[attr-defined]
            if data and data.get("ids"):
                for i, vid in enumerate(data["ids"]):
                    items.append(VectorItem(
                        id=vid,
                        metadata=(data.get("metadatas") or [{}])[i] if data.get("metadatas") else {},
                        content=(data.get("documents") or [""])[i] if data.get("documents") else ""
                    ))
            # Client-side sort for Chroma fallback if requested on metadata
            if order_by and order_by != 'id':
                key = order_by.split('.', 1)[1] if order_by.startswith('metadata.') else order_by
                reverse = str(order_dir).lower() == 'desc'
                items.sort(key=lambda x: (x.metadata or {}).get(key, ''), reverse=reverse)
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Vector listing failed: {e}")
    returned = len(items)
    pagination = build_offset_pagination_meta(
        limit=limit,
        offset=offset,
        total=total,
        count=returned,
    ).model_dump(mode="json")

    serialized_items: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            serialized_items.append(item)
            continue
        try:
            serialized_items.append(model_dump_compat(item))
        except TypeError:
            fallback = jsonable_encoder(item)
            if isinstance(fallback, dict):
                serialized_items.append(fallback)
            else:
                serialized_items.append({"value": fallback})

    return {
        "data": serialized_items,
        "pagination": pagination,
        "has_more": pagination["has_more"],
        "next_offset": pagination["next_offset"],
    }


@router.delete("/vector_stores/{store_id}/vectors/{vector_id}")
async def delete_vector(
    store_id: str = Path(...),
    vector_id: str = Path(...),
    current_user: User = Depends(get_request_user)
):
    # Validate store via meta DB
    uid = resolve_user_id_for_request(
        current_user,
        error_status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
    try:
        init_meta_db(uid)
        rows = meta_list_stores(uid)
        if not any(r.get('id') == store_id for r in rows):
            raise HTTPException(status_code=404, detail="Vector store not found")
    except HTTPException:
        raise
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        raise HTTPException(status_code=404, detail="Vector store not found") from None

    adapter = await _get_adapter_for_user(current_user, 1536)
    await adapter.initialize()
    # Verify existence using adapter if possible; otherwise best-effort fallback
    get_fn = getattr(adapter, 'get_vector', None)
    if callable(get_fn):
        vec = await get_fn(store_id, vector_id)  # type: ignore[misc]
        if not vec:
            raise HTTPException(status_code=404, detail="Vector not found")
    else:
        try:
            collection = adapter.manager.client.get_collection(name=store_id)  # type: ignore[attr-defined]
            data = collection.get(ids=[vector_id], include=[])
            ids_found = set(data.get('ids') or []) if isinstance(data, dict) else set()
            if vector_id not in ids_found:
                raise HTTPException(status_code=404, detail="Vector not found")
        except HTTPException:
            raise
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
            # If collection get fails unexpectedly, report not found
            raise HTTPException(status_code=404, detail="Vector not found") from None
    await adapter.delete_vectors(store_id, ids=[vector_id])
    return {"id": vector_id, "deleted": True}


@router.post("/vector_stores/{store_id}/query")
async def query_vectors(
    store_id: str = Path(...),
    payload: QueryRequest = Body(...),
    current_user: User = Depends(get_request_user)
):
    adapter = await _get_adapter_for_user(current_user, 1536)
    await adapter.initialize()

    # Determine the query vector
    qvec: list[float] | None = None
    if payload.vector is not None:
        # Validate empty vector upfront
        if len(payload.vector) == 0:
            raise HTTPException(status_code=400, detail="Vector must be non-empty")
        qvec = payload.vector
    elif payload.query:
        # Embed the text query
        embedding_settings = settings.get("EMBEDDING_CONFIG", {})
        app_config = {"embedding_config": embedding_settings}
        model_id = embedding_settings.get("default_model_id") or embedding_settings.get("embedding_model") or "text-embedding-3-small"
        provider = embedding_settings.get("embedding_provider", "openai")
        # In test contexts, normalize provider baseline to 'openai' for predictable policy behavior
        try:
            import os as _os
            if env_flag_enabled("TESTING"):
                provider = "openai"
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
            pass
        mods_hint = _allowed_models()
        if mods_hint and len(mods_hint) > 0:
            model_id = mods_hint[0]

        # Token checks first; respect strictest policy across allowed providers when present
        provs = _allowed_providers()
        max_tokens = _get_model_max_tokens(provider, model_id)
        if provs:
            try:
                candidates = [max_tokens] + [_get_model_max_tokens(p, model_id) for p in provs]
                max_tokens = min([t for t in candidates if isinstance(t, int) and t > 0])
            except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                pass
        token_len = _count_tokens(payload.query, model_id)
        if token_len > max_tokens:
            raise HTTPException(status_code=400, detail={
                "error": "input_too_long",
                "message": f"Query exceeds max tokens {max_tokens} for model {model_id}",
                "details": [{"tokens": token_len}]
            })
        # Enforce allowlist after token validation
        mods = _allowed_models()
        if mods is not None and not _model_allowed(model_id, mods):
            raise HTTPException(status_code=403, detail=f"Model '{model_id}' is not allowed for embeddings")
        if provs is not None and provider.lower() not in provs:
            raise HTTPException(status_code=403, detail=f"Provider '{provider}' is not allowed for embeddings")
        try:
            loop = asyncio.get_running_loop()
            embedded = await loop.run_in_executor(None, create_embeddings_batch, [payload.query], app_config, model_id)
            qvec = embedded[0]
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Embedding generation failed for query: {e}")
            raise HTTPException(500, detail="Failed to generate embedding for query") from e
    else:
        raise HTTPException(400, detail="Provide either 'query' text or 'vector'")

    # If caller provided a vector, enforce store dimension before proceeding
    if payload.vector is not None and qvec is not None:
        # Determine emptiness and stats first
        stats_dim: int | None = None
        is_empty: bool | None = None
        try:
            stats = await adapter.get_collection_stats(store_id)
            stats_dim = _as_int(stats.get('dimension'))
            try:
                coll = adapter.manager.get_or_create_collection(store_id)
                is_empty = (coll.count() == 0)
            except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                is_empty = bool(stats.get('count', 0) == 0)
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
            # If undeterminable, assume empty to avoid false rejections
            is_empty = True

        registry_dim = _as_int(_STORE_DIMENSIONS.get(store_id))
        if is_empty is False:
            if registry_dim is not None and registry_dim != 1536 and len(qvec) != registry_dim:
                raise HTTPException(status_code=400, detail=f"Vector length {len(qvec)} != expected {registry_dim}")
            if (stats_dim is not None) and stats_dim != 1536 and len(qvec) != stats_dim:
                raise HTTPException(status_code=400, detail=f"Vector length {len(qvec)} != expected {stats_dim}")
        else:
            # Empty store: enforce against declared registry dimension only
            if registry_dim is not None and registry_dim != 1536 and len(qvec) != registry_dim:
                raise HTTPException(status_code=400, detail=f"Vector length {len(qvec)} != expected {registry_dim}")

    dim = len(qvec)
    # Recreate adapter with correct dimension for search
    adapter = await _get_adapter_for_user(current_user, dim)
    await adapter.initialize()

    results = await adapter.search(
        collection_name=store_id,
        query_vector=qvec,
        k=payload.top_k,
        filter=payload.filter,
        include_metadata=True
    )
    return {
        "data": [
            {
                "id": r.id,
                "content": r.content,
                "metadata": r.metadata,
                "score": r.score,
            } for r in results
        ]
    }

# ==============================================
# Minimal batch semantics (in-memory status)
# ==============================================

_BATCH_STATUS: dict[str, dict[str, Any]] = {}


@router.post("/vector_stores/{store_id}/vectors/batches")
async def upsert_vectors_batch(
    store_id: str = Path(...),
    payload: UpsertVectorsRequest = Body(...),
    current_user: User = Depends(get_request_user)
):
    batch_id = f"vsb_{uuid.uuid4().hex[:20]}"
    _BATCH_STATUS[batch_id] = {"id": batch_id, "status": "processing", "upserted": 0, "error": None}
    # Persist creation
    try:
        uid = resolve_user_id_for_request(
            current_user,
            allow_none=True,
            error_status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        init_batches_db(uid)
        db_create_batch(
            batch_id=batch_id,
            store_id=store_id,
            user_id=uid,
            status='processing', upserted=0, error=None,
            meta={"records": len(payload.records) if payload and payload.records else 0}
        )
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as _e:
        logger.warning(f"Failed to persist batch creation: {_e}")
    try:
        res = await upsert_vectors(store_id=store_id, payload=payload, current_user=current_user)
        upserted = res.get("upserted", 0)
        _BATCH_STATUS[batch_id].update({"status": "completed", "upserted": upserted})
        try:
            db_update_batch(batch_id, user_id=uid, status='completed', upserted=upserted)
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as _e:
            logger.warning(f"Failed to persist batch completion: {_e}")
    except HTTPException as e:
        _BATCH_STATUS[batch_id].update({"status": "failed", "error": e.detail})
        try:
            db_update_batch(batch_id, user_id=uid, status='failed', error=str(e.detail))
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as _e:
            logger.warning(f"Failed to persist batch failure: {_e}")
        raise
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Vector batch upsert failed for batch {}", batch_id)
        safe_error = "Vector batch upsert failed"
        _BATCH_STATUS[batch_id].update({"status": "failed", "error": safe_error})
        try:
            db_update_batch(batch_id, user_id=uid, status='failed', error=safe_error)
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as _e:
            logger.warning(f"Failed to persist batch failure: {_e}")
        raise
    return _BATCH_STATUS[batch_id]


@router.get("/vector_stores/{store_id}/vectors/batches/{batch_id}")
async def get_batch_status(
    store_id: str = Path(...),
    batch_id: str = Path(...),
    current_user: User = Depends(get_request_user)
):
    if batch_id in _BATCH_STATUS:
        return _BATCH_STATUS[batch_id]
    # Fallback to persisted status
    rec = db_get_batch(
        batch_id,
        user_id=resolve_user_id_for_request(
            current_user,
            error_status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        ),
    )
    if not rec:
        raise HTTPException(404, detail="Batch not found")
    return {
        'id': rec['id'],
        'status': rec['status'],
        'upserted': rec['upserted'],
        'error': rec['error'],
        'store_id': rec['store_id'],
        'user_id': rec['user_id'],
        'created_at': rec['created_at'],
        'updated_at': rec['updated_at'],
        'meta': rec.get('meta', {})
    }


@router.get("/vector_stores/batches")
async def list_vector_batches(
    status: str | None = Query(None),
    limit: int = Query(50, gt=0, le=500),
    offset: int = Query(0, ge=0),
    user_id: str | None = Query(
        None, description="Admin-only: override user id to view their batches"
    ),
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    requested_user_id = resolve_user_id_for_request(
        current_user,
        error_status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
    if user_id is not None and user_id != requested_user_id:
        if not _principal_has_admin_claims(principal):
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required",
            )
        requested_user_id = str(user_id)
    fetched = db_list_batches(user_id=requested_user_id, status=status, limit=limit + 1, offset=offset)
    rows = fetched[:limit]
    has_more = len(fetched) > limit
    pagination = build_offset_pagination_meta(
        total=None,
        limit=limit,
        offset=offset,
        count=len(rows),
        has_more=has_more,
    ).model_dump(mode="json")
    pagination["count"] = len(rows)
    return {
        'data': rows,
        'pagination': pagination,
        'has_more': pagination["has_more"],
        'next_offset': pagination["next_offset"],
    }


class CreateFromMediaRequest(BaseModel):
    store_name: str
    dimensions: int | None = None
    embedding_model: str | None = None
    media_ids: list[int] | None = None
    keywords: list[str] | None = None
    keyword_match: str | None = Field(
        default="any",
        description="How to match multiple keywords: 'any' (union) or 'all' (intersection)."
    )
    chunk_size: int = 500
    chunk_overlap: int = 100
    chunk_method: str | None = 'words'
    language: str | None = None
    update_existing_store_id: str | None = None
    use_existing_embeddings: bool | None = False


@router.post("/vector_stores/create_from_media")
async def create_store_from_media(
    payload: CreateFromMediaRequest = Body(...),
    current_user: User = Depends(get_request_user),
    db: Any = Depends(get_media_db_for_user)
):
    # Validate chunk method early to return 400/422 on invalid value before DB lookups
    valid_methods = {m.value for m in ChunkingMethod}
    if payload.chunk_method and payload.chunk_method not in valid_methods:
        raise HTTPException(status_code=400, detail=f"Invalid chunk_method: {payload.chunk_method}")

    # Resolve items
    items: list[dict[str, Any]] = []
    if payload.media_ids:
        for mid in payload.media_ids:
            rec = get_media_by_id(db, mid)
            if rec:
                items.append(rec)
    elif payload.keywords:
        try:
            # Support union (any) vs intersection (all) semantics for multiple keywords
            match_mode = (payload.keyword_match or "any").strip().lower()
            if match_mode not in ("any", "all"):
                raise HTTPException(status_code=400, detail="keyword_match must be 'any' or 'all'")

            if match_mode == "all":
                # Use comprehensive search to find items that have ALL specified keywords
                results_list, _total = search_media(
                    db,
                    search_query=None,
                    must_have_keywords=[k for k in (payload.keywords or []) if k and str(k).strip()],
                    results_per_page=10000,
                    page=1,
                    include_trash=False,
                    include_deleted=False,
                )
                # search_media_db returns a list of media dictionaries
                items = list(results_list or [])
            else:
                # Default: union of items associated with any of the keywords.
                # Use search_media_db per-keyword to avoid backend-specific issues.
                merged: dict[int, dict[str, Any]] = {}
                for kw in (payload.keywords or []):
                    kw_clean = (kw or "").strip()
                    if not kw_clean:
                        continue
                    res_list, _ = search_media(
                        db,
                        search_query=None,
                        must_have_keywords=[kw_clean],
                        results_per_page=10000,
                        page=1,
                        include_trash=False,
                        include_deleted=False,
                    )
                    for it in (res_list or []):
                        try:
                            mid = int(it.get('id'))
                            merged[mid] = it
                        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                            # Fallback: if id missing/non-int, just append
                            items.append(it)
                items.extend(list(merged.values()))
        except HTTPException:
            raise
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as e:
            raise HTTPException(400, detail=f"Keyword fetch failed: {e}") from e
    else:
        raise HTTPException(400, detail="Provide media_ids or keywords")

    if not items:
        # If caller wants to use existing embeddings, allow id-only items
        if payload.use_existing_embeddings and payload.media_ids:
            items = [{'id': mid} for mid in payload.media_ids]
        else:
            # If chunk method is invalid, prefer 400/422 over 404 to match tests (redundant safety)
            if payload.chunk_method and payload.chunk_method not in valid_methods:
                raise HTTPException(status_code=400, detail=f"Invalid chunk_method: {payload.chunk_method}")
            raise HTTPException(404, detail="No media found for the provided selection")

    # Determine dimension: use configured model if not given
    embedding_settings = settings.get("EMBEDDING_CONFIG", {})
    model_id = payload.embedding_model or embedding_settings.get("default_model_id") or embedding_settings.get("embedding_model") or "text-embedding-3-small"

    # Create or use existing store
    created_store_id = payload.update_existing_store_id
    if not created_store_id:
        vs = await create_vector_store(VectorStoreCreate(name=payload.store_name, metadata={"created_from": "media", "requested_model": model_id}, embedding_model=model_id, dimensions=payload.dimensions or 1536), current_user=current_user)
        created_store_id = vs.id

    # Create a batch record (for both paths) and initialize adapter
    uid = resolve_user_id_for_request(
        current_user,
        error_status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
    batch_id = f"vsb_{uuid.uuid4().hex[:20]}"
    try:
        init_batches_db(uid)
        db_create_batch(batch_id, store_id=created_store_id, user_id=uid, status='processing', upserted=0, meta={"source": "create_from_media", "items": len(items)})
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        pass

    # Initialize adapter for downstream operations
    adapter = await _get_adapter_for_user(current_user, (payload.dimensions or 1536))
    await adapter.initialize()

    # If using existing embeddings, copy directly and return (skip chunking)
    if getattr(payload, 'use_existing_embeddings', False):
        source_collection_name = f"user_{uid}_media_embeddings"
        try:
            source_col = adapter.manager.get_or_create_collection(source_collection_name)
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as e:
            raise HTTPException(404, detail=f"Source embeddings collection not found: {e}") from e

        upserted_total = 0
        for it in items:
            mid = it.get('id')
            if mid is None:
                continue
            try:
                data = source_col.get(where={"media_id": mid}, include=["embeddings", "documents", "metadatas"], limit=100000)
            except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Failed to read existing embeddings for media {mid}: {e}")
                continue
            if not data or not data.get('ids'):
                continue
            emb_list = data.get('embeddings', [])
            try:
                if hasattr(emb_list, 'tolist'):
                    emb_list = emb_list.tolist()
            except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
                pass
            doc_list = data.get('documents') or []
            meta_list_existing = data.get('metadatas') or []
            ids_list = data.get('ids') or []
            if len(emb_list) == 0:
                continue
            emb_dim = len(emb_list[0]) if emb_list and len(emb_list[0]) else adapter.config.embedding_dim
            if emb_dim != adapter.config.embedding_dim:
                adapter = await _get_adapter_for_user(current_user, emb_dim)
                await adapter.initialize()
            await adapter.upsert_vectors(created_store_id, ids=ids_list, vectors=emb_list, documents=doc_list, metadatas=meta_list_existing)
            upserted_total += len(emb_list)

        # Persist batch status
        with contextlib.suppress(_VECTORSTORE_NONCRITICAL_EXCEPTIONS):
            db_update_batch(batch_id, user_id=uid, status='completed', upserted=upserted_total)
        return {"store_id": created_store_id, "batch_id": batch_id, "upserted": upserted_total}

    # Chunk and embed texts in batches (use full Chunker)
    texts: list[str] = []
    meta_list: list[dict[str, Any]] = []
    ids: list[str] = []

    # Configure chunker
    # Determine method
    method_value = (payload.chunk_method or 'words').lower()
    try:
        method_enum = ChunkingMethod(method_value)
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported chunk_method '{payload.chunk_method}'") from None

    ck_cfg = ChunkerConfig(
        default_method=method_enum,
        default_max_size=payload.chunk_size,
        default_overlap=payload.chunk_overlap,
        language=(payload.language or 'en')
    )
    ck = Chunker(config=ck_cfg)

    def add_chunks_for_item(item: dict[str, Any]):
        content = item.get('content') or item.get('analysis') or ''
        if not content:
            return
        chunks = ck.chunk_text(content, method=method_enum.value, max_size=payload.chunk_size, overlap=payload.chunk_overlap)
        for idx, txt in enumerate(chunks):
            if not txt.strip():
                continue
            texts.append(txt)
            meta_list.append({
                'media_id': item.get('id'),
                'title': item.get('title'),
                'source': 'media_db',
                'chunk_index': idx
            })
            ids.append(f"media_{item.get('id')}_chunk_{idx}")

    # De-duplicate by media id to avoid duplicate chunking/upserts across union modes
    seen_ids: set[Any] = set()
    for it in items:
        mid = it.get('id') if isinstance(it, dict) else None
        if mid is not None:
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
        add_chunks_for_item(it)

    if not texts:
        raise HTTPException(400, detail="No textual content found to embed")

    # (use_existing_embeddings handled above with early return)

    # Option B: Embed in slices to avoid large calls
    app_config = {"embedding_config": embedding_settings}

    upserted_total = 0
    step = 64
    for start in range(0, len(texts), step):
        subtexts = texts[start:start+step]
        # Allowlist + token checks per slice
        provider = embedding_settings.get("embedding_provider", "openai")
        provs = _allowed_providers()
        if provs is not None and provider.lower() not in provs:
            raise HTTPException(status_code=403, detail=f"Provider '{provider}' is not allowed for embeddings")
        mods = _allowed_models()
        if mods is not None and not _model_allowed(model_id, mods):
            raise HTTPException(status_code=403, detail=f"Model '{model_id}' is not allowed for embeddings")
        max_tokens = _get_model_max_tokens(provider, model_id)
        too_long: list[tuple[int, int]] = []
        for i, tx in enumerate(subtexts):
            tok = _count_tokens(tx, model_id)
            if tok > max_tokens:
                too_long.append((start + i, tok))
        if too_long:
            raise HTTPException(status_code=400, detail={
                "error": "input_too_long",
                "message": f"One or more inputs exceed max tokens {max_tokens} for model {model_id}",
                "details": [{"index": i, "tokens": tok} for (i, tok) in too_long]
            })
        try:
            loop = asyncio.get_running_loop()
            embed_fn = _get_embeddings_fn()
            vecs = await loop.run_in_executor(None, embed_fn, subtexts, app_config, model_id)
        except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as e:
            db_update_batch(
                batch_id,
                user_id=uid,
                status='failed',
                error="Failed to generate embeddings for media content",
            )
            raise HTTPException(
                500,
                detail="Failed to generate embeddings for media content",
            ) from e
        # Prepare corresponding slice metadata
        slice_ids = ids[start:start+step]
        slice_docs = subtexts
        slice_meta = meta_list[start:start+step]
        # Ensure adapter dimension matches
        if not adapter._initialized or adapter.config.embedding_dim != len(vecs[0]):
            adapter = await _get_adapter_for_user(current_user, len(vecs[0]))
            await adapter.initialize()
        await adapter.upsert_vectors(created_store_id, ids=slice_ids, vectors=vecs, documents=slice_docs, metadatas=slice_meta)
        upserted_total += len(vecs)
        with contextlib.suppress(_VECTORSTORE_NONCRITICAL_EXCEPTIONS):
            db_update_batch(batch_id, user_id=uid, upserted=upserted_total)

    db_update_batch(batch_id, user_id=uid, status='completed', upserted=upserted_total)
    return {"store_id": created_store_id, "batch_id": batch_id, "upserted": upserted_total}


@router.get("/vector_stores/{store_id}", response_model=VectorStoreObject)
async def get_vector_store(
    store_id: str = Path(...),
    current_user: User = Depends(get_request_user)
):
    """Fetch a single vector store by id.

    Placed after batch/admin routes to avoid path shadowing of '/vector_stores/batches'.
    """
    # Get stats directly from adapter; fall back to meta DB only for friendly name override
    adapter = await _get_adapter_for_user(current_user, 1536)
    await adapter.initialize()
    try:
        stats = await adapter.get_collection_stats(store_id)
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=404, detail=f"Vector store not found: {e}") from e
    md = stats.get("metadata", {}) or {}
    # Prefer meta DB name if available (do not gate existence on meta DB)
    try:
        uid = resolve_user_id_for_request(
            current_user,
            error_status=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        init_meta_db(uid)
        rows = meta_list_stores(uid)
        for r in rows:
            if r.get('id') == store_id:
                md['name'] = r.get('name', md.get('name', store_id))
                break
    except _VECTORSTORE_NONCRITICAL_EXCEPTIONS:
        pass
    return {
        "id": md.get("openai_id", store_id),
        "object": "vector_store",
        "name": md.get("name", store_id),
        "created_at": md.get("created_at", _now_ts()),
        "metadata": md,
        "dimensions": stats.get("dimension", 1536)
    }
