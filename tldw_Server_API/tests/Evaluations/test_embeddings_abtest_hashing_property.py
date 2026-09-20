import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.api.v1.schemas.embeddings_abtest_schemas import (
    EmbeddingsABTestConfig,
)
from tldw_Server_API.app.core.Evaluations.embeddings_abtest_service import (
    _compute_collection_hash,
    _compute_pipeline_hash,
)

_TEXT_ALPHABET = string.ascii_letters + string.digits + "/_-"


def _dump_config(config: EmbeddingsABTestConfig) -> dict:
    if hasattr(config, "model_dump"):
        return config.model_dump()
    return config.dict()


def _build_config(payload: dict) -> EmbeddingsABTestConfig:
    if hasattr(EmbeddingsABTestConfig, "model_validate"):
        return EmbeddingsABTestConfig.model_validate(payload)  # type: ignore[attr-defined]
    return EmbeddingsABTestConfig.parse_obj(payload)  # type: ignore[attr-defined]


@st.composite
def abtest_config(draw):
    provider = draw(st.sampled_from(["openai", "huggingface", "cohere", "mistral"]))
    model = draw(st.text(alphabet=_TEXT_ALPHABET, min_size=3, max_size=24))
    dimensions = draw(st.one_of(st.none(), st.integers(min_value=4, max_value=2048)))
    media_ids = draw(st.lists(st.integers(min_value=1, max_value=5000), min_size=0, max_size=5))

    chunking = {
        "method": draw(st.sampled_from(["sentences", "words", "paragraphs"])),
        "size": draw(st.integers(min_value=50, max_value=2000)),
        "overlap": draw(st.integers(min_value=0, max_value=200)),
        "language": draw(st.one_of(st.none(), st.sampled_from(["en", "es", "fr"]))),
    }

    re_ranker = draw(
        st.one_of(
            st.none(),
            st.fixed_dictionaries(
                {
                    "provider": st.sampled_from(["flashrank", "cohere"]),
                    "model": st.text(
                        alphabet=_TEXT_ALPHABET,
                        min_size=3,
                        max_size=24,
                    ),
                }
            ),
        )
    )

    retrieval = {
        "k": draw(st.integers(min_value=1, max_value=999)),
        "search_mode": draw(st.sampled_from(["vector", "hybrid", "fts"])),
        "hybrid_alpha": draw(
            st.one_of(
                st.none(),
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            )
        ),
        "re_ranker": re_ranker,
        "apply_reranker": draw(st.booleans()),
    }

    query_texts = draw(st.lists(st.text(alphabet=_TEXT_ALPHABET, min_size=1, max_size=40), min_size=1, max_size=3))
    queries = [{"text": query_text} for query_text in query_texts]

    return {
        "arms": [
            {"provider": provider, "model": model, "dimensions": dimensions}
        ],
        "media_ids": media_ids,
        "chunking": chunking,
        "retrieval": retrieval,
        "queries": queries,
        "metric_level": draw(st.sampled_from(["media", "chunk"])),
        "reuse_existing": True,
    }


@pytest.mark.unit
@given(abtest_config())
@settings(max_examples=25)
def test_collection_hash_deterministic(config):
    config = _build_config(config)
    assert _compute_collection_hash(config, 0) == _compute_collection_hash(config, 0)


@pytest.mark.unit
@given(abtest_config())
@settings(max_examples=25)
def test_collection_hash_media_order_invariant(config):
    config = _build_config(config)
    payload = _dump_config(config)
    payload["media_ids"] = list(reversed(payload["media_ids"]))
    reordered = _build_config(payload)
    assert _compute_collection_hash(config, 0) == _compute_collection_hash(reordered, 0)


@pytest.mark.unit
@given(abtest_config())
@settings(max_examples=25)
def test_collection_hash_changes_on_chunking_size(config):
    config = _build_config(config)
    payload = _dump_config(config)
    payload["chunking"]["size"] = int(payload["chunking"]["size"]) + 1
    mutated = _build_config(payload)
    assert _compute_collection_hash(config, 0) != _compute_collection_hash(mutated, 0)


@pytest.mark.unit
@given(abtest_config())
@settings(max_examples=25)
def test_pipeline_hash_deterministic(config):
    config = _build_config(config)
    assert _compute_pipeline_hash(config) == _compute_pipeline_hash(config)


@pytest.mark.unit
@given(abtest_config())
@settings(max_examples=25)
def test_pipeline_hash_changes_on_k(config):
    config = _build_config(config)
    payload = _dump_config(config)
    payload["retrieval"]["k"] = int(payload["retrieval"]["k"]) + 1
    mutated = _build_config(payload)
    assert _compute_pipeline_hash(config) != _compute_pipeline_hash(mutated)
