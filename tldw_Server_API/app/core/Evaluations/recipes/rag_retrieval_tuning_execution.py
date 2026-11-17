"""Execution-planning helpers for the RAG retrieval tuning recipe."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from tldw_Server_API.app.core.RAG.rag_schemas_unified import UnifiedRAGRequest


@dataclass(frozen=True)
class CandidateIndexPlan:
    """Isolated index planning for a retrieval candidate."""

    candidate_id: str
    index_key: str
    base_index_key: str
    indexing_signature: str
    needs_rebuild: bool
    mutates_live_index: bool = False
    reuse_index_key: str | None = None


def plan_candidate_indexes(
    *,
    corpus_scope: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    dataset_content_hash: str,
    owner_user_id: str,
) -> dict[str, CandidateIndexPlan]:
    """Plan isolated index namespaces for retrieval candidates."""
    base_index_key = _build_base_index_key(
        corpus_scope=corpus_scope,
        dataset_content_hash=dataset_content_hash,
        owner_user_id=owner_user_id,
    )
    plans: dict[str, CandidateIndexPlan] = {}
    first_index_key_by_signature: dict[str, str] = {}
    seen_candidate_ids: set[str] = set()

    for position, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate.get("candidate_id") or f"candidate-{position}").strip()
        if candidate_id in seen_candidate_ids:
            raise ValueError(f"candidate_id values must be unique; duplicate {candidate_id!r}")
        seen_candidate_ids.add(candidate_id)
        indexing_config = dict(candidate.get("indexing_config") or {})
        indexing_signature = _build_indexing_signature(indexing_config)
        index_affecting = _is_index_affecting(indexing_config)
        reuse_index_key = first_index_key_by_signature.get(indexing_signature)

        if reuse_index_key is not None:
            index_key = reuse_index_key
            needs_rebuild = False
        elif index_affecting:
            index_key = _build_candidate_index_key(
                base_index_key=base_index_key,
                candidate_id=candidate_id,
                indexing_signature=indexing_signature,
            )
            needs_rebuild = True
        else:
            index_key = base_index_key
            needs_rebuild = False

        if reuse_index_key is None:
            first_index_key_by_signature[indexing_signature] = index_key

        plans[candidate_id] = CandidateIndexPlan(
            candidate_id=candidate_id,
            index_key=index_key,
            base_index_key=base_index_key,
            indexing_signature=indexing_signature,
            needs_rebuild=needs_rebuild,
            mutates_live_index=False,
            reuse_index_key=reuse_index_key,
        )

    return plans


def build_unified_rag_request(
    *,
    query: str,
    corpus_scope: Mapping[str, Any],
    candidate: Mapping[str, Any],
    index_key: str | None = None,
) -> UnifiedRAGRequest:
    """Map a recipe candidate and corpus scope into the unified RAG request schema."""
    retrieval_config = dict(candidate.get("retrieval_config") or {})
    request_payload: dict[str, Any] = {
        "query": query,
        "sources": list(corpus_scope.get("sources") or ["media_db"]),
        "include_media_ids": _coerce_media_ids(corpus_scope.get("media_ids")),
        "include_note_ids": _coerce_note_ids(corpus_scope.get("note_ids")),
        "search_mode": retrieval_config.get("search_mode", "hybrid"),
        "top_k": retrieval_config.get("top_k", 10),
        "hybrid_alpha": retrieval_config.get("hybrid_alpha", 0.7),
        "enable_reranking": retrieval_config.get("enable_reranking", False),
        "reranking_strategy": retrieval_config.get("reranking_strategy", "none"),
        "rerank_top_k": retrieval_config.get("rerank_top_k"),
    }
    if index_key:
        request_payload["index_namespace"] = index_key
    return UnifiedRAGRequest(**request_payload)


def summarize_candidate_metrics(
    *,
    first_pass_hits: list[Mapping[str, Any]],
    reranked_hits: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize first-pass retrieval and post-rerank quality separately."""
    first_pass_grades = _extract_grades(first_pass_hits)
    reranked_grades = _extract_grades(reranked_hits) or first_pass_grades
    pre_rerank_recall_at_k = _normalized_grade_average(first_pass_grades)
    post_rerank_ndcg_at_k = _ndcg(reranked_grades)

    return {
        "first_pass_recall_score": pre_rerank_recall_at_k,
        "post_rerank_quality_score": post_rerank_ndcg_at_k,
        "metrics": {
            "pre_rerank_recall_at_k": pre_rerank_recall_at_k,
            "post_rerank_ndcg_at_k": post_rerank_ndcg_at_k,
            "first_pass_hit_count": len(first_pass_grades),
            "post_rerank_hit_count": len(reranked_grades),
        },
    }


def serialize_index_plan(plan: CandidateIndexPlan) -> dict[str, Any]:
    """Convert a dataclass execution plan into a plain dictionary."""
    return asdict(plan)


def _build_base_index_key(
    *,
    corpus_scope: Mapping[str, Any],
    dataset_content_hash: str,
    owner_user_id: str,
) -> str:
    scope_payload = {
        "owner_user_id": owner_user_id,
        "dataset_content_hash": dataset_content_hash,
        "sources": list(corpus_scope.get("sources") or []),
        "media_ids": list(corpus_scope.get("media_ids") or []),
        "note_ids": list(corpus_scope.get("note_ids") or []),
    }
    scope_signature = _hash_payload(scope_payload)
    return f"rag-eval/{owner_user_id}/{scope_signature[:16]}"


def _build_candidate_index_key(
    *,
    base_index_key: str,
    candidate_id: str,
    indexing_signature: str,
) -> str:
    return f"{base_index_key}/{candidate_id}-{indexing_signature[:12]}"


def _build_indexing_signature(indexing_config: Mapping[str, Any]) -> str:
    if not indexing_config:
        return "shared"
    return _hash_payload(dict(indexing_config))


def _hash_payload(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _is_index_affecting(indexing_config: Mapping[str, Any]) -> bool:
    if not indexing_config:
        return False
    chunking_preset = str(indexing_config.get("chunking_preset") or "").strip().lower()
    return chunking_preset not in {"", "fixed_index"}


def _coerce_media_ids(raw_ids: Any) -> list[int] | None:
    if raw_ids is None:
        return None
    values = [int(item) for item in list(raw_ids)]
    return values or None


def _coerce_note_ids(raw_ids: Any) -> list[str] | None:
    if raw_ids is None:
        return None
    values = [str(item).strip() for item in list(raw_ids) if str(item).strip()]
    return values or None


def _extract_grades(hits: list[Mapping[str, Any]]) -> list[int]:
    grades: list[int] = []
    for hit in hits:
        try:
            grades.append(min(3, max(0, int(hit.get("grade", 0)))))
        except (TypeError, ValueError):
            grades.append(0)
    return grades


def _normalized_grade_average(grades: list[int]) -> float:
    if not grades:
        return 0.0
    return round(sum(grades) / (len(grades) * 3), 6)


def _ndcg(grades: list[int]) -> float:
    if not grades:
        return 0.0
    dcg = 0.0
    for position, grade in enumerate(grades, start=1):
        dcg += (2**grade - 1) / math.log2(position + 1)
    ideal_grades = sorted(grades, reverse=True)
    ideal_dcg = 0.0
    for position, grade in enumerate(ideal_grades, start=1):
        ideal_dcg += (2**grade - 1) / math.log2(position + 1)
    if ideal_dcg == 0.0:
        return 0.0
    return round(dcg / ideal_dcg, 6)
