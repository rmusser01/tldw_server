from __future__ import annotations

import pytest

import tldw_Server_API.app.core.Embeddings.vector_processing as vector_processing_module
from tldw_Server_API.app.core.Embeddings.request_types import EmbeddingProviderError
from tldw_Server_API.app.core.Embeddings.vector_processing import (
    EmbeddingVectorProcessor,
)


@pytest.mark.unit
def test_validate_vector_count_canonicalizes_finite_numeric_vectors():
    result = EmbeddingVectorProcessor().validate_vector_count(
        [[1, 2], [3.5, 4]],
        expected=2,
        provider="p",
        model="m",
    )

    assert result == [[1.0, 2.0], [3.5, 4.0]]
    assert all(isinstance(value, float) for vector in result for value in vector)


@pytest.mark.unit
def test_validate_vector_count_preserves_exact_count_mismatch_error():
    with pytest.raises(EmbeddingProviderError) as exc_info:
        EmbeddingVectorProcessor().validate_vector_count(
            [[1.0]],
            expected=2,
            provider="p",
            model="m",
        )

    error = exc_info.value
    assert error.code == "provider_malformed_response"
    assert error.message == "Embedding provider returned 1 embeddings, expected 2"
    assert error.provider == "p"
    assert error.model == "m"


@pytest.mark.unit
def test_validate_vector_count_preserves_exact_same_count_malformed_error():
    with pytest.raises(EmbeddingProviderError) as exc_info:
        EmbeddingVectorProcessor().validate_vector_count(
            [[1.0, 2.0], [3.0]],
            expected=2,
            provider="p",
            model="m",
        )

    error = exc_info.value
    assert error.code == "provider_malformed_response"
    assert error.message == "Embedding provider returned malformed embedding vectors"
    assert error.provider == "p"
    assert error.model == "m"


@pytest.mark.unit
@pytest.mark.parametrize(
    "vectors",
    [
        pytest.param([[True, 1.0]], id="bool"),
        pytest.param([[float("nan"), 1.0]], id="nan"),
        pytest.param([[float("inf"), 1.0]], id="infinite"),
        pytest.param([["not-a-number", 1.0]], id="nonnumeric"),
    ],
)
def test_validate_vector_count_rejects_malformed_numeric_values(vectors):
    with pytest.raises(EmbeddingProviderError) as exc_info:
        EmbeddingVectorProcessor().validate_vector_count(
            vectors,
            expected=1,
            provider="p",
            model="m",
        )

    error = exc_info.value
    assert error.code == "provider_malformed_response"
    assert error.message == "Embedding provider returned malformed embedding vectors"
    assert error.provider == "p"
    assert error.model == "m"


@pytest.mark.unit
@pytest.mark.parametrize(
    "vector",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="empty"),
        pytest.param((1.0, 2.0), id="non-list"),
        pytest.param(["not-a-number"], id="nonnumeric"),
        pytest.param([True, 0.0], id="bool"),
        pytest.param([float("nan"), 0.0], id="nan"),
        pytest.param([float("inf"), 0.0], id="infinite"),
    ],
)
def test_validate_cached_vector_returns_none_for_nullable_or_malformed_values(vector):
    assert EmbeddingVectorProcessor().validate_cached_vector(vector) is None


@pytest.mark.unit
def test_validate_cached_vector_returns_canonical_finite_vector():
    result = EmbeddingVectorProcessor().validate_cached_vector([1, 2.5])

    assert result == [1.0, 2.5]
    assert result is not None
    assert all(isinstance(value, float) for value in result)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dimensions", "dimension_policy", "expected", "expected_calls"),
    [
        pytest.param(2, "reduce", [[1.0, 2.0]], [("p", "m", "reduce")], id="reduce"),
        pytest.param(
            4,
            "pad",
            [[1.0, 2.0, 3.0, 0.0]],
            [("p", "m", "pad")],
            id="pad",
        ),
        pytest.param(None, "reduce", [[1.0, 2.0, 3.0]], [], id="no-dimensions"),
    ],
)
def test_process_vectors_applies_dimensions_and_records_adjustments(
    dimensions,
    dimension_policy,
    expected,
    expected_calls,
):
    calls: list[tuple[str, str, str]] = []
    processor = EmbeddingVectorProcessor(record_dimension_adjustment=lambda *args: calls.append(args))

    result = processor.process_vectors(
        [[1, 2, 3]],
        provider="p",
        model="m",
        dimensions=dimensions,
        dimension_policy=dimension_policy,
    )

    assert result == expected
    assert all(isinstance(value, float) for vector in result for value in vector)
    assert calls == expected_calls


@pytest.mark.unit
def test_process_vectors_canonicalizes_before_and_after_dimension_adjustment(monkeypatch):
    def adjust_dimensions_probe(vectors, target_dim, provider, model, **kwargs):
        assert vectors == [[1.0, 2.0]]
        assert all(isinstance(value, float) for value in vectors[0])
        assert (target_dim, provider, model) == (2, "p", "m")
        assert kwargs["dimension_policy"] == "reduce"
        return [[3, 4]]

    monkeypatch.setattr(
        vector_processing_module,
        "adjust_dimensions",
        adjust_dimensions_probe,
    )

    result = EmbeddingVectorProcessor().process_vectors(
        [[1, 2]],
        provider="p",
        model="m",
        dimensions=2,
        dimension_policy="reduce",
    )

    assert result == [[3.0, 4.0]]
    assert all(isinstance(value, float) for value in result[0])


@pytest.mark.unit
def test_process_cached_vector_postprocesses_an_already_canonical_vector():
    processor = EmbeddingVectorProcessor()
    canonical = processor.validate_cached_vector([1, 2, 3])
    assert canonical is not None

    result = processor.process_cached_vector(
        canonical,
        provider="p",
        model="m",
        dimensions=2,
        dimension_policy="reduce",
    )

    assert result == [1.0, 2.0]


@pytest.mark.unit
def test_process_vectors_propagates_recorder_exception_identity():
    original = RuntimeError("recorder failed")

    def raise_original(provider: str, model: str, method: str) -> None:
        del provider, model, method
        raise original

    processor = EmbeddingVectorProcessor(record_dimension_adjustment=raise_original)

    with pytest.raises(RuntimeError) as exc_info:
        processor.process_vectors(
            [[1.0, 2.0, 3.0]],
            provider="p",
            model="m",
            dimensions=2,
            dimension_policy="reduce",
        )

    assert exc_info.value is original
