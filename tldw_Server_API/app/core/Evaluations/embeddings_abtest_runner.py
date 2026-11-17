from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from tldw_Server_API.app.core.Evaluations.embeddings_abtest_schemas import (
    EmbeddingsABTestConfig,
)


async def _embed_texts(
    provider: str,
    model: str,
    texts: list[str],
    metadata: dict[str, Any] | None = None,
) -> list[list[float]]:
    """Create embeddings for texts using the enhanced embeddings endpoint utilities.

    Returns L2-normalized vectors for numeric outputs.
    """
    from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import (
        create_embeddings_batch_async,
    )

    embeddings = await create_embeddings_batch_async(
        texts=texts,
        provider=provider,
        model_id=model,
        metadata=metadata,
    )
    # L2-normalize to ensure consistent scoring if caller relies on numeric vectors
    normed: list[list[float]] = []
    for vec in embeddings:
        arr = np.array(vec, dtype=np.float32)
        nrm = float(np.linalg.norm(arr))
        if nrm > 0:
            arr = arr / nrm
        normed.append(arr.tolist())
    return normed


async def prepare_query_embeddings_for_arms(config: EmbeddingsABTestConfig) -> dict[str, list[list[float]]]:
    """Embed all queries per arm using each arm's provider/model.

    This utility helps run vector-only comparisons without modifying retrievers.
    Returns a mapping of arm_key ("provider:model") -> list of query vectors.
    """
    texts = [q.text for q in config.queries]
    results: dict[str, list[list[float]]] = {}

    for arm in config.arms:
        arm_key = f"{arm.provider}:{arm.model}"
        logger.info(f"Embedding {len(texts)} queries for arm {arm_key}")
        try:
            vecs = await _embed_texts(arm.provider, arm.model, texts)
            results[arm_key] = vecs
        except Exception as e:
            logger.error(f"Failed to embed queries for arm {arm_key}: {e}")
            results[arm_key] = []
    return results
