from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from typing import Any

import numpy as np
from loguru import logger

from tldw_Server_API.app.core.Evaluations.embeddings_abtest_schemas import (
    EmbeddingsABTestConfig,
)
from tldw_Server_API.app.core.Chunking import Chunker, ChunkerConfig
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import MediaDbLike
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
from tldw_Server_API.app.core.Evaluations.embeddings_abtest_jobs import (
    ABTEST_JOBS_CLEANUP_TYPE,
    ABTEST_JOBS_DOMAIN,
    abtest_jobs_idempotency_key,
    abtest_jobs_manager,
    abtest_jobs_queue,
)
from tldw_Server_API.app.core.Evaluations.embeddings_abtest_metrics import (
    record_abtest_arm_build,
    record_abtest_run,
)
from tldw_Server_API.app.core.Evaluations.metrics_retrieval import (
    hit_at_k,
    mrr,
    ndcg,
    recall_at_k,
)
from tldw_Server_API.app.core.RAG.rag_service.advanced_reranking import RerankingStrategy
from tldw_Server_API.app.core.testing import env_flag_enabled

_ABTEST_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    json.JSONDecodeError,
)


class EmbeddingsABTestRunError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = True, backoff_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        if backoff_seconds is not None:
            self.backoff_seconds = backoff_seconds


class EmbeddingsABTestPolicyError(EmbeddingsABTestRunError):
    def __init__(
        self,
        message: str,
        *,
        policy_type: str,
        status_code: int,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, retryable=False)
        self.policy_type = policy_type
        self.status_code = status_code
        self.details = details or {}


def _parse_abtest_quota(raw: object) -> int | None:
    if raw is None:
        return None
    try:
        value = int(str(raw).strip())
    except _ABTEST_NONCRITICAL_EXCEPTIONS:
        return None
    return value if value > 0 else None


def _load_abtest_quota(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        try:
            from tldw_Server_API.app.core.config import settings as app_settings
            raw = app_settings.get(name)
        except _ABTEST_NONCRITICAL_EXCEPTIONS:
            raw = None
    return _parse_abtest_quota(raw)


def _resolve_abtest_embedding_config(app_settings: dict[str, Any]) -> dict[str, Any]:
    """Build embedding config for A/B runs with test-safe Chroma defaults."""
    raw = app_settings.get("EMBEDDING_CONFIG", {})
    embedding_config = raw.copy() if isinstance(raw, dict) else {}

    base_dir = app_settings.get("USER_DB_BASE_DIR")
    if not base_dir:
        with contextlib.suppress(_ABTEST_NONCRITICAL_EXCEPTIONS):
            base_dir = str(DatabasePaths.get_user_db_base_dir(allow_legacy_alias=True))
    if not base_dir:
        base_dir = "Databases/user_databases"
    embedding_config["USER_DB_BASE_DIR"] = str(base_dir)

    chroma_settings = embedding_config.get("chroma_client_settings")
    if not isinstance(chroma_settings, dict):
        chroma_settings = {}
    try:
        testing_mode = env_flag_enabled("TESTING")
    except _ABTEST_NONCRITICAL_EXCEPTIONS:
        testing_mode = False
    if testing_mode:
        chroma_settings["use_in_memory_stub"] = True
    embedding_config["chroma_client_settings"] = chroma_settings
    return embedding_config


def _model_allowed(model: str, allowed_models: list[str]) -> bool:
    for pat in allowed_models:
        if pat.endswith("*") and model.startswith(pat[:-1]):
            return True
        if model == pat:
            return True
    return False


def validate_abtest_policy(config: EmbeddingsABTestConfig, *, user: object | None = None) -> None:
    try:
        from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import (
            _get_allowed_models,
            _get_allowed_providers,
            _should_enforce_policy,
        )
    except _ABTEST_NONCRITICAL_EXCEPTIONS:
        def _get_allowed_providers():
            return None
        def _get_allowed_models():
            return None
        def _should_enforce_policy(_user=None):
            return False

    enforce_policy = bool(_should_enforce_policy(user))
    allowed_providers = _get_allowed_providers()
    allowed_models = _get_allowed_models()

    if enforce_policy and allowed_providers is not None:
        for arm in config.arms:
            provider = (arm.provider or "").lower()
            if provider not in allowed_providers:
                raise EmbeddingsABTestPolicyError(
                    f"Provider '{arm.provider}' is not allowed",
                    policy_type="provider",
                    status_code=403,
                    details={"provider": arm.provider, "allowed_providers": allowed_providers},
                )

    if enforce_policy and allowed_models is not None:
        for arm in config.arms:
            if not _model_allowed(arm.model, allowed_models):
                raise EmbeddingsABTestPolicyError(
                    f"Model '{arm.model}' is not allowed",
                    policy_type="model",
                    status_code=403,
                    details={"model": arm.model, "allowed_models": allowed_models},
                )

    quotas = {
        "arms": (len(config.arms), "EVALS_ABTEST_MAX_ARMS"),
        "queries": (len(config.queries), "EVALS_ABTEST_MAX_QUERIES"),
        "media_ids": (len(config.media_ids), "EVALS_ABTEST_MAX_MEDIA_IDS"),
    }
    for label, (count, env_key) in quotas.items():
        limit = _load_abtest_quota(env_key)
        if limit is None:
            continue
        if count > limit:
            raise EmbeddingsABTestPolicyError(
                f"A/B test exceeds {label} quota ({count} > {limit})",
                policy_type="quota",
                status_code=429,
                details={"quota": env_key, "count": count, "limit": limit},
            )


def _compute_collection_hash(config: EmbeddingsABTestConfig, arm_index: int) -> str:
    arm = config.arms[arm_index]
    payload = {
        "media_ids": sorted(config.media_ids),
        "chunking": {
            "method": config.chunking.method,
            "size": config.chunking.size,
            "overlap": config.chunking.overlap,
            "language": config.chunking.language,
        },
        "arm": {"provider": arm.provider, "model": arm.model, "dimensions": arm.dimensions},
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _compute_pipeline_hash(config: EmbeddingsABTestConfig) -> str:
    payload = {
        "retrieval": {
            "k": config.retrieval.k,
            "search_mode": config.retrieval.search_mode,
            "hybrid_alpha": config.retrieval.hybrid_alpha,
            "re_ranker": (
                {
                    "provider": config.retrieval.re_ranker.provider,
                    "model": config.retrieval.re_ranker.model,
                }
                if config.retrieval.re_ranker
                else None
            ),
        },
        "metric_level": config.metric_level,
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


async def _embed_texts(
    provider: str,
    model: str,
    texts: list[str],
    metadata: dict[str, Any] | None = None,
) -> list[list[float]]:
    from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import (
        create_embeddings_batch_async,
    )
    vectors = await create_embeddings_batch_async(
        texts=texts,
        provider=provider,
        model_id=model,
        metadata=metadata,
    )
    out: list[list[float]] = []
    for v in vectors:
        arr = np.array(v, dtype=np.float32)
        nrm = float(np.linalg.norm(arr))
        if nrm > 0:
            arr = arr / nrm
        out.append(arr.tolist())
    return out


def _get_model_revision(provider: str, model: str) -> dict[str, str | None]:
    meta: dict[str, str | None] = {"hf_revision": None, "onnx_sha": None}
    try:
        if provider.lower() == "huggingface":
            from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import COMMIT_HASHES
            meta["hf_revision"] = COMMIT_HASHES.get(model)
        elif provider.lower() == "onnx":
            from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import COMMIT_HASHES
            meta["onnx_sha"] = COMMIT_HASHES.get(model)
    except _ABTEST_NONCRITICAL_EXCEPTIONS:
        pass
    return meta


def _collection_exists(manager: ChromaDBManager, name: str) -> bool:
    try:
        collections = manager.list_collections()
    except _ABTEST_NONCRITICAL_EXCEPTIONS:
        return False
    for coll in collections:
        if isinstance(coll, dict):
            if coll.get("name") == name:
                return True
        else:
            if getattr(coll, "name", None) == name:
                return True
    return False


def get_collection_manager(
    *,
    user_id: str,
    embedding_config: dict[str, Any],
    manager_factory: Any | None = None,
) -> ChromaDBManager:
    """Build the collection manager through an injectable seam for tests."""
    if manager_factory is not None:
        return manager_factory(
            user_id=str(user_id),
            embedding_config=embedding_config,
        )
    return ChromaDBManager(
        user_id=str(user_id),
        user_embedding_config=embedding_config,
    )


async def build_collections_vector_only(
    db: EvaluationsDatabase,
    config: EmbeddingsABTestConfig,
    test_id: str,
    user_id: str,
    media_db: MediaDbLike,
    *,
    manager_factory: Any | None = None,
) -> list[dict[str, str]]:
    """Chunk, embed, and store vectors per arm into per-user collections.

    Returns list of {arm_id, collection_name} for each arm.
    """
    from tldw_Server_API.app.core.config import settings as app_settings
    embedding_config = _resolve_abtest_embedding_config(app_settings)

    manager = get_collection_manager(
        user_id=str(user_id),
        embedding_config=embedding_config,
        manager_factory=manager_factory,
    )

    results: list[dict[str, str]] = []
    pipeline_hash = _compute_pipeline_hash(config)
    existing_arms = {}
    for arm in db.get_abtest_arms(test_id) or []:
        try:
            idx = arm.get("arm_index")
        except AttributeError:
            continue
        if idx is None:
            continue
        try:
            existing_arms[int(idx)] = arm
        except (TypeError, ValueError):
            continue

    # Prepare chunker
    cconf = ChunkerConfig(
        default_method=config.chunking.method,
        default_max_size=config.chunking.size,
        default_overlap=config.chunking.overlap,
        language=config.chunking.language or "en",
    )
    chunker = Chunker(config=cconf)

    test_row = db.get_abtest(test_id) or {}
    created_by = test_row.get("created_by")

    # Load corpus content
    corpus_texts: list[tuple[int, str]] = []  # (media_id, text)
    for mid in config.media_ids:
        item = media_db.get_media_by_id(int(mid))
        if not item or not item.get("content"):
            continue
        text = item.get("content")
        if isinstance(text, dict):
            text = text.get("content", "")
        corpus_texts.append((int(mid), str(text or "")))

    # Test-mode fallback: if no media_ids provided, synthesize a tiny corpus
    # from the queries themselves so reranking logic can be exercised.
    try:
        if not corpus_texts and env_flag_enabled("TESTING"):
            # Use index-based synthetic media IDs to avoid collisions
            corpus_texts = [(100000 + i, q.text) for i, q in enumerate(config.queries or [])]
    except _ABTEST_NONCRITICAL_EXCEPTIONS:
        pass

    for i, arm in enumerate(config.arms):
        collection_name = f"user_{user_id}_abtest_{test_id}_arm_{i}"
        collection_hash = _compute_collection_hash(config, i)
        arm_logger = logger.bind(
            test_id=test_id,
            arm_index=i,
            provider=arm.provider,
            model=arm.model,
        )
        arm_start = time.monotonic()
        arm_logger.info("Embeddings A/B collection build starting")
        existing = existing_arms.get(i)
        reuse_arm = None
        reuse_collection_name = collection_name
        shared_origin_test_id = None

        if (
            config.reuse_existing
            and existing
            and existing.get("collection_hash") == collection_hash
            and existing.get("collection_name") == collection_name
            and existing.get("status") == "ready"
            and _collection_exists(manager, collection_name)
        ):
            # Reuse only within the same test lifecycle (collection name is test-scoped).
            reuse_arm = existing
        elif config.reuse_existing and created_by:
            reuse_candidate = db.find_reusable_abtest_arm(
                test_id=test_id,
                collection_hash=collection_hash,
                created_by=created_by,
            )
            if (
                reuse_candidate
                and reuse_candidate.get("collection_hash") == collection_hash
                and reuse_candidate.get("status") == "ready"
            ):
                candidate_name = reuse_candidate.get("collection_name")
                if candidate_name and _collection_exists(manager, candidate_name):
                    reuse_arm = reuse_candidate
                    reuse_collection_name = candidate_name
                    shared_origin_test_id = reuse_candidate.get("test_id")

        if reuse_arm:
            stats_payload = None
            meta_payload = None
            try:
                if reuse_arm.get("stats_json"):
                    stats_payload = json.loads(reuse_arm.get("stats_json") or "{}")
            except _ABTEST_NONCRITICAL_EXCEPTIONS:
                stats_payload = None
            try:
                if reuse_arm.get("metadata_json"):
                    meta_payload = json.loads(reuse_arm.get("metadata_json") or "{}")
            except _ABTEST_NONCRITICAL_EXCEPTIONS:
                meta_payload = None
            if shared_origin_test_id:
                if meta_payload is None:
                    meta_payload = {}
                meta_payload["shared_collection"] = True
                meta_payload["shared_origin_test_id"] = shared_origin_test_id
                try:
                    origin_meta = meta_payload if shared_origin_test_id == test_id else None
                    if shared_origin_test_id != test_id:
                        origin_meta = {}
                        if reuse_arm.get("metadata_json"):
                            origin_meta = json.loads(reuse_arm.get("metadata_json") or "{}")
                        shared_with = set(origin_meta.get("shared_with") or [])
                        shared_with.add(test_id)
                        origin_meta["shared_collection"] = True
                        origin_meta["shared_with"] = sorted(shared_with)
                        origin_stats = None
                        try:
                            if reuse_arm.get("stats_json"):
                                origin_stats = json.loads(reuse_arm.get("stats_json") or "{}")
                        except _ABTEST_NONCRITICAL_EXCEPTIONS:
                            origin_stats = None
                        db.upsert_abtest_arm(
                            test_id=reuse_arm.get("test_id"),
                            arm_index=int(reuse_arm.get("arm_index") or 0),
                            provider=reuse_arm.get("provider") or arm.provider,
                            model_id=reuse_arm.get("model_id") or arm.model,
                            dimensions=reuse_arm.get("dimensions"),
                            collection_hash=reuse_arm.get("collection_hash"),
                            pipeline_hash=reuse_arm.get("pipeline_hash"),
                            collection_name=reuse_arm.get("collection_name"),
                            status=reuse_arm.get("status") or "ready",
                            stats_json=origin_stats,
                            metadata_json=origin_meta,
                        )
                except _ABTEST_NONCRITICAL_EXCEPTIONS:
                    pass

            arm_id = db.upsert_abtest_arm(
                test_id=test_id,
                arm_index=i,
                provider=arm.provider,
                model_id=arm.model,
                dimensions=reuse_arm.get("dimensions") or arm.dimensions,
                collection_hash=collection_hash,
                pipeline_hash=pipeline_hash,
                collection_name=reuse_collection_name,
                status='ready',
                stats_json=stats_payload,
                metadata_json=meta_payload,
            )
            arm_logger = arm_logger.bind(arm_id=arm_id)
            results.append({"arm_id": arm_id, "collection_name": reuse_collection_name})
            with contextlib.suppress(_ABTEST_NONCRITICAL_EXCEPTIONS):
                record_abtest_arm_build(
                    duration_seconds=time.monotonic() - arm_start,
                    status="reused",
                    provider=arm.provider,
                    model=arm.model,
                )
            arm_logger.info("Embeddings A/B collection reused")
            continue

        arm_id = db.upsert_abtest_arm(
            test_id=test_id,
            arm_index=i,
            provider=arm.provider,
            model_id=arm.model,
            dimensions=arm.dimensions,
            collection_hash=collection_hash,
            pipeline_hash=pipeline_hash,
            status='building',
        )
        arm_logger = arm_logger.bind(arm_id=arm_id)

        try:
            try:
                if _collection_exists(manager, collection_name):
                    manager.delete_collection(collection_name)
            except _ABTEST_NONCRITICAL_EXCEPTIONS:
                pass

            # Chunk entire corpus
            all_texts: list[str] = []
            metadatas: list[dict[str, str]] = []
            ids: list[str] = []
            for (mid, text) in corpus_texts:
                chunks = chunker.chunk_text_with_metadata(
                    text=text,
                    method=config.chunking.method,
                    max_size=config.chunking.size,
                    overlap=config.chunking.overlap,
                )
                for idx, ch in enumerate(chunks):
                    all_texts.append(ch.text)
                    ids.append(f"mid{mid}_ch{idx}")
                    md = {
                        "media_id": str(mid),
                        "chunk_index": str(idx),
                        "chunk_start": str(getattr(ch.metadata, 'start_char', idx * (config.chunking.size - config.chunking.overlap))),
                        "chunk_end": str(getattr(ch.metadata, 'end_char', (idx + 1) * config.chunking.size)),
                        "embedding_model": arm.model,
                        "embedding_provider": arm.provider,
                    }
                    md.update(_get_model_revision(arm.provider, arm.model))
                    metadatas.append(md)

            # Embed and store
            user_metadata = {"user_id": str(user_id)}
            vectors = await _embed_texts(arm.provider, arm.model, all_texts, metadata=user_metadata) if all_texts else []
            emb_dim = len(vectors[0]) if vectors else (arm.dimensions or 0)

            # store in chroma
            if vectors:
                manager.store_in_chroma(
                    texts=all_texts,
                    embeddings=vectors,
                    ids=ids,
                    metadatas=metadatas,
                    collection_name=collection_name,
                )

            # Update DB with collection info and metadata
            db.upsert_abtest_arm(
                test_id=test_id,
                arm_index=i,
                provider=arm.provider,
                model_id=arm.model,
                dimensions=emb_dim or arm.dimensions,
                collection_hash=collection_hash,
                pipeline_hash=pipeline_hash,
                collection_name=collection_name,
                status='ready',
                stats_json={
                    "chunk_count": len(all_texts),
                    "embedding_dim": emb_dim,
                    "doc_count": len(corpus_texts)
                },
                metadata_json={"hf_revision": _get_model_revision(arm.provider, arm.model).get("hf_revision"),
                               "onnx_sha": _get_model_revision(arm.provider, arm.model).get("onnx_sha")},
            )

            results.append({"arm_id": arm_id, "collection_name": collection_name})
            with contextlib.suppress(_ABTEST_NONCRITICAL_EXCEPTIONS):
                record_abtest_arm_build(
                    duration_seconds=time.monotonic() - arm_start,
                    status="built",
                    provider=arm.provider,
                    model=arm.model,
                )
            arm_logger.info("Embeddings A/B collection build completed")
        except _ABTEST_NONCRITICAL_EXCEPTIONS as exc:
            with contextlib.suppress(_ABTEST_NONCRITICAL_EXCEPTIONS):
                db.upsert_abtest_arm(
                    test_id=test_id,
                    arm_index=i,
                    provider=arm.provider,
                    model_id=arm.model,
                    dimensions=arm.dimensions,
                    collection_hash=collection_hash,
                    pipeline_hash=pipeline_hash,
                    collection_name=collection_name,
                    status='failed',
                    stats_json={"error": str(exc)},
                )
            with contextlib.suppress(_ABTEST_NONCRITICAL_EXCEPTIONS):
                record_abtest_arm_build(
                    duration_seconds=time.monotonic() - arm_start,
                    status="failed",
                    provider=arm.provider,
                    model=arm.model,
                )
            arm_logger.warning(f"Embeddings A/B collection build failed: {exc}")
            raise EmbeddingsABTestRunError(
                f"Failed to build collection for arm {arm.provider}/{arm.model}: {exc}",
                retryable=True,
            ) from exc

    return results


async def run_vector_search_and_score(
    db: EvaluationsDatabase,
    config: EmbeddingsABTestConfig,
    test_id: str,
    user_id: str,
    arm_collections: list[dict[str, str]],
    *,
    manager_factory: Any | None = None,
) -> dict[str, dict[str, float]]:
    """Run vector-only search across arms and compute metrics, storing results in DB.

    Returns aggregate metrics per arm_id.
    """
    search_mode = (config.retrieval.search_mode or "vector").lower()
    if search_mode == "fts":
        raise EmbeddingsABTestRunError(
            "fts search_mode is not supported by the embeddings A/B vector retrieval path.",
            retryable=False,
        )

    from tldw_Server_API.app.core.config import settings as app_settings
    embedding_config = _resolve_abtest_embedding_config(app_settings)
    manager = get_collection_manager(
        user_id=str(user_id),
        embedding_config=embedding_config,
        manager_factory=manager_factory,
    )

    # Embed queries per arm
    # Ensure DB has queries and align IDs
    qrows = db.get_abtest_queries(test_id)
    if not qrows:
        # fallback to config-only
        texts = [q.text for q in config.queries]
        qids = [f"q_{i}" for i in range(len(texts))]
        gt_lookup = {qids[i]: [str(x) for x in (config.queries[i].expected_ids or [])] for i in range(len(texts))}
    else:
        texts = [r.get('text','') for r in qrows]
        qids = [r.get('query_id') for r in qrows]
        def _parse_ids(s: str | None) -> list[str]:
            if not s:
                return []
            try:
                v = json.loads(s)
                return [str(x) for x in (v or [])]
            except _ABTEST_NONCRITICAL_EXCEPTIONS:
                return []
        gt_lookup = {r.get('query_id'): _parse_ids(r.get('ground_truth_ids')) for r in qrows}
    query_vecs_per_arm: dict[str, list[list[float]]] = {}
    # Use sequential arm order
    query_metadata = {"user_id": str(user_id)}
    for i, arm in enumerate(config.arms):
        key = f"arm_{test_id}_{i}"
        query_vecs_per_arm[key] = await _embed_texts(arm.provider, arm.model, texts, metadata=query_metadata)

    # Run searches and score
    aggregates: dict[str, dict[str, float]] = {}
    metric_level = config.metric_level or "media"
    include_media_ids = config.media_ids or None
    for i, mapping in enumerate(arm_collections):
        arm_id = mapping["arm_id"]
        collection_name = mapping["collection_name"]
        k = config.retrieval.k
        qvecs = query_vecs_per_arm.get(arm_id) or query_vecs_per_arm.get(f"arm_{test_id}_{i}")
        if not qvecs:
            continue

        per_query_scores = {"recall": [], "mrr": [], "ndcg": [], "hit": [], "latency_ms": []}
        for q_idx, qid in enumerate(qids):
            start = time.time()
            ranked: list[str] = []
            distances: list[list[float]] = [[]]
            metadatas: list[list[dict[str, Any]]] = [[]]
            documents: list[list[str]] = [[]]
            rerank_scores_out: list[float] | None = None
            if search_mode == 'hybrid':
                try:
                    from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
                    result = await unified_rag_pipeline(
                        query=texts[q_idx],
                        search_mode='hybrid',
                        hybrid_alpha=float(config.retrieval.hybrid_alpha or 0.7),
                        top_k=k,
                        index_namespace=collection_name,
                        user_id=str(user_id),
                        include_media_ids=include_media_ids,
                    )
                    docs = result.documents or []
                    for d in docs:
                        md = d.metadata if isinstance(getattr(d, "metadata", None), dict) else {}
                        doc_id = getattr(d, "id", None)
                        ranked.append(str(doc_id) if doc_id is not None else str(md.get("media_id") or ""))
                        metadatas[0].append(md if isinstance(md, dict) else {})
                        documents[0].append(str(getattr(d, "content", "")))
                        score = getattr(d, "score", None)
                        distances[0].append(1.0 - float(score) if score is not None else 0.0)
                except _ABTEST_NONCRITICAL_EXCEPTIONS as e:
                    logger.error(f"Hybrid pipeline failed for {collection_name}: {e}")
                    continue
            else:
                qvec = qvecs[q_idx]
                collection = manager.get_or_create_collection(collection_name)
                ranked: list[str] = []
                try:
                    # Chroma include: valid keys are documents, embeddings, metadatas, distances, uris, data
                    # 'ids' are returned by default and not a valid include key on some versions.
                    res = collection.query(
                        query_embeddings=[qvec],
                        n_results=k,
                        include=["documents", "metadatas", "distances"]
                    )
                    ids = res.get("ids") or [[]]
                    metadatas = res.get("metadatas") or [[]]
                    documents = res.get("documents") or [[]]
                    distances = res.get("distances") or [[]]
                    ranked = [str(x) for x in (ids[0] if ids else [])]
                except _ABTEST_NONCRITICAL_EXCEPTIONS as e:
                    # Fallback: no results but still proceed, so toggle-on rerank can persist baseline scores
                    logger.warning(f"Vector search failed for {collection_name}; proceeding with empty results: {e}")
                # Optional rerank controlled by toggle
                if getattr(config.retrieval, 're_ranker', None) and bool(getattr(config.retrieval, 'apply_reranker', False)):
                    from tldw_Server_API.app.core.RAG.rag_service.advanced_reranking import (
                        RerankingConfig,
                        create_reranker,
                    )
                    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource
                    from tldw_Server_API.app.core.RAG.rag_service.types import Document as RagDocument

                    def _map_strategy(provider: str | None, model: str | None) -> RerankingStrategy:
                        p = (provider or '').lower()
                        m = (model or '').lower()
                        # Explicit FlashRank
                        if 'flashrank' in p or 'flashrank' in m:
                            return RerankingStrategy.FLASHRANK
                        # Cross-encoder cues
                        cross_cues = [
                            'cross_encoder', 'cross-encoder', 'crossencoder',
                            'mono', 'monot5', 't5', 'ms-marco', 'msmarco',
                            'bge-reranker', 'gte-reranker', 'reranker', 're-rank', 'rerank',
                            'cohere', 'voyage', 'nv-rerank'
                        ]
                        if any(c in p for c in cross_cues) or any(c in m for c in cross_cues):
                            return RerankingStrategy.CROSS_ENCODER
                        # LLM scoring cues
                        llm_cues = ['llm', 'gpt', 'claude', 'sonnet', 'haiku', 'mistral', 'mixtral', 'gemini', 'qwen', 'command']
                        if any(c in p for c in llm_cues) or any(c in m for c in llm_cues):
                            return RerankingStrategy.LLM_SCORING
                        # Diversity/MMR
                        if 'mmr' in p or 'mmr' in m or 'diversity' in p or 'diversity' in m:
                            return RerankingStrategy.DIVERSITY
                        # Hybrid/multi
                        if 'hybrid' in p or 'hybrid' in m or 'multi' in p or 'multi' in m:
                            return RerankingStrategy.HYBRID
                        # Default
                        return RerankingStrategy.FLASHRANK

                    # Build documents and baseline scores up-front so we always record rerank_scores when toggled on
                    docs: list[RagDocument] = []
                    orig_scores: list[float] = []
                    for i2 in range(len(ranked)):
                        md = metadatas[0][i2] if metadatas and metadatas[0] else {}
                        content = documents[0][i2] if documents and documents[0] else ""
                        score = 1.0 - float(distances[0][i2]) if distances and distances[0] else 0.0
                        d = RagDocument(id=ranked[i2], content=content, metadata=md, source=DataSource.MEDIA_DB, score=score)
                        docs.append(d)
                        orig_scores.append(score)
                    # Default to original scores if reranker fails for any reason
                    # Ensure non-empty scores when reranker is toggled on, to surface presence in exports
                    rerank_scores_out = list(orig_scores) if orig_scores else [0.0]

                    try:
                        rr = config.retrieval.re_ranker
                        strat = _map_strategy(getattr(rr, 'provider', None), getattr(rr, 'model', None)) if rr else RerankingStrategy.FLASHRANK
                        rconf = RerankingConfig(strategy=strat, top_k=k, model_name=(getattr(rr, 'model', None) if rr else None))
                        reranker = create_reranker(strat, rconf)
                        scored = await reranker.rerank(query=texts[q_idx], documents=docs, original_scores=orig_scores)
                        new_ranked: list[str] = []
                        new_scores: list[float] = []
                        for sd in scored:
                            mid = None
                            md = sd.document.metadata if hasattr(sd.document, 'metadata') else None
                            if isinstance(md, dict):
                                mid = md.get('media_id')
                            new_ranked.append(str(mid) if mid is not None else str(sd.document.id))
                            new_scores.append(float(getattr(sd, 'rerank_score', 0.0)))
                        ranked = new_ranked
                        # overwrite distances with normalized 1 - score ordering surrogate
                        if new_scores:
                            distances = [[1.0 - s for s in new_scores]]
                            rerank_scores_out = list(new_scores)
                    except _ABTEST_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(f"Reranking failed; using original ordering: {e}")
            elapsed = (time.time() - start) * 1000.0

            def _parse_media_id_from_chunk_id(rid: str) -> str | None:
                if not rid.startswith("mid"):
                    return None
                try:
                    head = rid.split("_", 1)[0]
                    return head[3:] if head.startswith("mid") else None
                except _ABTEST_NONCRITICAL_EXCEPTIONS:
                    return None

            ranked_media_ids: list[str] = []
            if metadatas and metadatas[0]:
                for idx, rid in enumerate(ranked):
                    md = metadatas[0][idx] if idx < len(metadatas[0]) else {}
                    mid = md.get("media_id") if isinstance(md, dict) else None
                    if mid is None and isinstance(rid, str):
                        mid = _parse_media_id_from_chunk_id(rid)
                    ranked_media_ids.append(str(mid) if mid is not None else str(rid))
            else:
                ranked_media_ids = [str(_parse_media_id_from_chunk_id(rid) or rid) for rid in ranked]

            ranked_for_metrics = ranked_media_ids if metric_level == "media" else ranked

            # Ground truth
            gt_ids = gt_lookup.get(qid, [])
            per_query_scores["recall"].append(recall_at_k(ranked_for_metrics, gt_ids, k))
            per_query_scores["mrr"].append(mrr(ranked_for_metrics, gt_ids, k))
            per_query_scores["ndcg"].append(ndcg(ranked_for_metrics, gt_ids, k))
            per_query_scores["hit"].append(hit_at_k(ranked_for_metrics, gt_ids, k))
            per_query_scores["latency_ms"].append(elapsed)

            # Store individual result row
            db.insert_abtest_result(
                test_id=test_id,
                arm_id=arm_id,
                query_id=qid,
                ranked_ids=ranked,
                scores=None,
                metrics={
                    "recall_at_k": per_query_scores["recall"][-1],
                    "mrr": per_query_scores["mrr"][-1],
                    "ndcg": per_query_scores["ndcg"][-1],
                    "hit_at_k": per_query_scores["hit"][-1],
                },
                latency_ms=elapsed,
                ranked_distances=(distances[0] if distances and distances[0] else None),
                ranked_metadatas=(metadatas[0] if metadatas and metadatas[0] else None),
                ranked_documents=(documents[0] if documents and documents[0] else None),
                rerank_scores=rerank_scores_out,
            )

        # Aggregate
        def _avg(xs: list[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        aggregates[arm_id] = {
            "recall_at_k": _avg(per_query_scores["recall"]),
            "mrr": _avg(per_query_scores["mrr"]),
            "ndcg": _avg(per_query_scores["ndcg"]),
            "hit_at_k": _avg(per_query_scores["hit"]),
            "latency_ms_p50": float(np.percentile(per_query_scores["latency_ms"], 50)) if per_query_scores["latency_ms"] else 0.0,
            "latency_ms_p95": float(np.percentile(per_query_scores["latency_ms"], 95)) if per_query_scores["latency_ms"] else 0.0,
            "latency_ms_mean": _avg(per_query_scores["latency_ms"]),
        }

    return aggregates


async def run_abtest_full(
    db: EvaluationsDatabase,
    config: EmbeddingsABTestConfig,
    test_id: str,
    user_id: str,
    media_db: MediaDbLike,
) -> None:
    """Background job that builds collections and executes evaluation, updating DB progress."""
    run_start = time.monotonic()
    run_logger = logger.bind(test_id=test_id)
    run_logger.info("Embeddings A/B test run starting")
    try:
        validate_abtest_policy(config)
        db.set_abtest_status(test_id, 'running', stats_json={"progress": {"phase": 0.05}})
        arm_info = await build_collections_vector_only(db, config, test_id, user_id, media_db)
        db.set_abtest_status(test_id, 'running', stats_json={"progress": {"phase": 0.5}})
        aggregates = await run_vector_search_and_score(db, config, test_id, user_id, arm_info)
        sig = compute_significance(db, test_id, metric='ndcg')
        db.set_abtest_status(test_id, 'completed', stats_json={"aggregates": aggregates, "significance": sig, "progress": {"phase": 1.0}})
        with contextlib.suppress(_ABTEST_NONCRITICAL_EXCEPTIONS):
            record_abtest_run(duration_seconds=time.monotonic() - run_start, status="completed")
        run_logger.info("Embeddings A/B test run completed")
        cleanup = getattr(config, "cleanup_policy", None)
        if cleanup and bool(getattr(cleanup, "on_complete", False)):
            try:
                cleanup_abtest_resources(db, user_id, test_id, delete_db=True, delete_idempotency=True)
            except _ABTEST_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"Failed to cleanup A/B test {test_id} after completion: {exc}")
        elif cleanup and getattr(cleanup, "ttl_hours", None):
            try:
                from datetime import datetime, timedelta, timezone

                ttl_hours = int(cleanup.ttl_hours)
                if ttl_hours > 0:
                    abtest_jobs_manager().create_job(
                        domain=ABTEST_JOBS_DOMAIN,
                        queue=abtest_jobs_queue(),
                        job_type=ABTEST_JOBS_CLEANUP_TYPE,
                        payload={"test_id": test_id, "user_id": user_id},
                        owner_user_id=str(user_id),
                        priority=5,
                        max_retries=1,
                        available_at=datetime.now(timezone.utc) + timedelta(hours=ttl_hours),
                        idempotency_key=abtest_jobs_idempotency_key(test_id, "cleanup"),
                    )
            except _ABTEST_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"Failed to enqueue cleanup job for A/B test {test_id}: {exc}")
    except EmbeddingsABTestRunError as exc:
        with contextlib.suppress(_ABTEST_NONCRITICAL_EXCEPTIONS):
            record_abtest_run(duration_seconds=time.monotonic() - run_start, status="failed")
        run_logger.warning(f"Embeddings A/B test run failed: {exc}")
        raise
    except _ABTEST_NONCRITICAL_EXCEPTIONS as exc:
        with contextlib.suppress(_ABTEST_NONCRITICAL_EXCEPTIONS):
            record_abtest_run(duration_seconds=time.monotonic() - run_start, status="failed")
        run_logger.warning(f"Embeddings A/B test run failed: {exc}")
        raise EmbeddingsABTestRunError(f"A/B test {test_id} failed: {exc}", retryable=True) from exc


def cleanup_abtest_resources(
    db: EvaluationsDatabase,
    user_id: str,
    test_id: str,
    *,
    delete_db: bool,
    delete_idempotency: bool,
    created_by: str | None = None,
    manager_factory: Any | None = None,
) -> dict[str, int]:
    from tldw_Server_API.app.core.config import settings as app_settings

    deleted = 0
    embedding_config = _resolve_abtest_embedding_config(app_settings)
    manager = get_collection_manager(
        user_id=str(user_id),
        embedding_config=embedding_config,
        manager_factory=manager_factory,
    )
    arms = db.get_abtest_arms(test_id, created_by=created_by)
    for arm in arms:
        cname = arm.get("collection_name")
        if not cname:
            continue
        try:
            meta = json.loads(arm.get("metadata_json") or "{}") if arm.get("metadata_json") else {}
        except _ABTEST_NONCRITICAL_EXCEPTIONS:
            meta = {}
        if isinstance(meta, dict) and meta.get("shared_collection"):
            continue
        try:
            manager.delete_collection(cname)
            deleted += 1
        except _ABTEST_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"Cleanup failed for collection {cname}: {exc}")
    db_deleted = 0
    if delete_db:
        try:
            db_deleted = int(db.delete_abtest(test_id, delete_idempotency=delete_idempotency, created_by=created_by))
        except _ABTEST_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"Cleanup failed for A/B test rows {test_id}: {exc}")
    return {"collections_deleted": deleted, "abtests_deleted": db_deleted}


def _sign_test_pvalue(wins: int, losses: int) -> float:
    """Compute two-sided sign test p-value using exact binomial tail (no SciPy)."""
    import math
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    # P = 2 * sum_{i=0..k} C(n, i) * (0.5)^n
    def comb(n, r):
        return math.comb(n, r)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2.0 * tail)
    return p


def compute_significance(db: EvaluationsDatabase, test_id: str, metric: str = 'ndcg') -> dict[str, dict[str, float]]:
    """Compute pairwise significance (sign test) over per-query metrics by arm.

    Returns nested dict {arm_i: {arm_j: p_value}}.
    """
    arms = db.get_abtest_arms(test_id)
    queries = db.get_abtest_queries(test_id)
    qids = [r['query_id'] for r in queries]
    # Build per-arm metrics per query
    per_arm: dict[str, dict[str, float]] = {a['arm_id']: {} for a in arms}
    # Fetch all results (could paginate if large)
    rows, _total = db.list_abtest_results(test_id, limit=100000, offset=0)
    for r in rows:
        arm_id = r['arm_id']
        qid = r['query_id']
        try:
            m = json.loads(r.get('metrics_json') or '{}')
            val = float(m.get(metric, 0.0))
            per_arm.setdefault(arm_id, {})[qid] = val
        except _ABTEST_NONCRITICAL_EXCEPTIONS:
            pass

    # Pairwise p-values
    pvals: dict[str, dict[str, float]] = {}
    for _i, ai in enumerate(arms):
        a_id = ai['arm_id']
        pvals[a_id] = {}
        for _j, aj in enumerate(arms):
            b_id = aj['arm_id']
            if a_id == b_id:
                pvals[a_id][b_id] = 1.0
                continue
            wins = losses = 0
            for q in qids:
                va = per_arm.get(a_id, {}).get(q)
                vb = per_arm.get(b_id, {}).get(q)
                if va is None or vb is None:
                    continue
                if va > vb:
                    wins += 1
                elif vb > va:
                    losses += 1
            pvals[a_id][b_id] = _sign_test_pvalue(wins, losses)
    return pvals
