from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Embeddings.input_normalizer import normalize_embedding_input
from tldw_Server_API.app.core.Embeddings.request_types import EmbeddingInputError, EmbeddingPolicyError


def _count_tokens(text: str, model: str) -> int:
    del model
    return len(text.split())


def _tokens_to_texts(tokens_input: list[int] | list[list[int]], model: str):
    del model
    if tokens_input and isinstance(tokens_input[0], int):
        return ["decoded-single"], len(tokens_input), [len(tokens_input)]
    return (
        [f"decoded-{idx}" for idx, _ in enumerate(tokens_input)],
        sum(len(item) for item in tokens_input),
        [len(item) for item in tokens_input],
    )


def _normalize(raw_input, *, max_tokens: int = 100):
    return normalize_embedding_input(
        raw_input,
        model="test-model",
        max_tokens=max_tokens,
        count_tokens=_count_tokens,
        tokens_to_texts=_tokens_to_texts,
    )


def _assert_input_error(exc_info, code: str, message: str, details: list[dict] | None = None) -> None:
    error = exc_info.value
    assert isinstance(error, EmbeddingInputError)
    assert error.code == code
    assert error.message == message
    assert error.details == (details or [])


@pytest.mark.unit
def test_string_input_normalizes_to_single_text_with_counted_tokens():
    normalized = _normalize("hello world")

    assert normalized.texts == ["hello world"]
    assert normalized.token_counts == [2]
    assert normalized.total_tokens == 2
    assert normalized.provided_token_arrays is False
    assert normalized.token_input_mode == "none"


@pytest.mark.unit
def test_list_of_strings_normalizes_and_rejects_empty_strings():
    normalized = _normalize(["hello world", "again"])

    assert normalized.texts == ["hello world", "again"]
    assert normalized.token_counts == [2, 1]
    assert normalized.total_tokens == 3
    assert normalized.provided_token_arrays is False
    assert normalized.token_input_mode == "none"

    with pytest.raises(EmbeddingInputError) as exc_info:
        _normalize(["hello", " "])

    _assert_input_error(exc_info, "empty_input", "Input list cannot contain empty strings")


@pytest.mark.unit
def test_empty_string_and_empty_list_errors_preserve_code_and_message():
    with pytest.raises(EmbeddingInputError) as exc_info:
        _normalize(" ")

    _assert_input_error(exc_info, "empty_input", "Input cannot be empty")

    with pytest.raises(EmbeddingInputError) as exc_info:
        _normalize([])

    _assert_input_error(exc_info, "empty_input", "Input list cannot be empty")


@pytest.mark.unit
def test_mixed_list_error_preserves_code_and_message():
    with pytest.raises(EmbeddingInputError) as exc_info:
        _normalize(["hello", 1])

    _assert_input_error(exc_info, "invalid_input_type", "Invalid input type")


@pytest.mark.unit
def test_list_size_over_2048_preserves_code_and_message():
    with pytest.raises(EmbeddingInputError) as exc_info:
        _normalize(["x"] * 2049)

    _assert_input_error(exc_info, "too_many_inputs", "Maximum 2048 inputs allowed")

    with pytest.raises(EmbeddingInputError) as exc_info:
        _normalize([[1]] * 2049)

    _assert_input_error(exc_info, "too_many_inputs", "Maximum 2048 inputs allowed")


@pytest.mark.unit
def test_single_token_array_uses_decoder_and_raw_token_length():
    normalized = _normalize([101, 102, 103])

    assert normalized.texts == ["decoded-single"]
    assert normalized.token_counts == [3]
    assert normalized.total_tokens == 3
    assert normalized.provided_token_arrays is True
    assert normalized.token_input_mode == "single"


@pytest.mark.unit
def test_batch_token_arrays_use_decoder_and_per_array_raw_lengths():
    normalized = _normalize([[101, 102], [201, 202, 203]])

    assert normalized.texts == ["decoded-0", "decoded-1"]
    assert normalized.token_counts == [2, 3]
    assert normalized.total_tokens == 5
    assert normalized.provided_token_arrays is True
    assert normalized.token_input_mode == "batch"


@pytest.mark.unit
def test_token_array_decode_failure_maps_to_invalid_token_array():
    def failing_tokens_to_texts(tokens_input, model):
        del tokens_input, model
        raise ValueError("decode failed")

    with pytest.raises(EmbeddingInputError) as exc_info:
        normalize_embedding_input(
            [1, 2, 3],
            model="test-model",
            max_tokens=100,
            count_tokens=_count_tokens,
            tokens_to_texts=failing_tokens_to_texts,
        )

    _assert_input_error(exc_info, "invalid_token_array", "Invalid token array input")


@pytest.mark.unit
def test_token_array_decoder_domain_error_propagates_without_invalid_input_mapping():
    expected_error = EmbeddingPolicyError("provider_denied", "Provider denied")

    def failing_tokens_to_texts(tokens_input, model):
        del tokens_input, model
        raise expected_error

    with pytest.raises(EmbeddingPolicyError) as exc_info:
        normalize_embedding_input(
            [1, 2, 3],
            model="test-model",
            max_tokens=100,
            count_tokens=_count_tokens,
            tokens_to_texts=failing_tokens_to_texts,
        )

    assert exc_info.value is expected_error


@pytest.mark.unit
def test_token_array_decoder_text_count_mismatch_maps_to_invalid_token_array():
    def mismatched_tokens_to_texts(tokens_input, model):
        del tokens_input, model
        return ["decoded-0"], 3, [1, 2]

    with pytest.raises(EmbeddingInputError) as exc_info:
        normalize_embedding_input(
            [[101], [201, 202]],
            model="test-model",
            max_tokens=100,
            count_tokens=_count_tokens,
            tokens_to_texts=mismatched_tokens_to_texts,
        )

    _assert_input_error(exc_info, "invalid_token_array", "Invalid token array input")


@pytest.mark.unit
def test_token_array_decoder_token_lengths_count_mismatch_maps_to_invalid_token_array():
    def mismatched_tokens_to_texts(tokens_input, model):
        del tokens_input, model
        return ["decoded-0", "decoded-1"], 3, [1]

    with pytest.raises(EmbeddingInputError) as exc_info:
        normalize_embedding_input(
            [[101], [201, 202]],
            model="test-model",
            max_tokens=100,
            count_tokens=_count_tokens,
            tokens_to_texts=mismatched_tokens_to_texts,
        )

    _assert_input_error(exc_info, "invalid_token_array", "Invalid token array input")


@pytest.mark.unit
@pytest.mark.parametrize(
    "provided_token_lengths",
    [
        3,
        "3",
        b"3",
        ["1"],
        [True],
        [-1],
    ],
)
def test_token_array_decoder_malformed_provided_token_lengths_maps_to_invalid_token_array(
    provided_token_lengths,
):
    def malformed_tokens_to_texts(tokens_input, model):
        del tokens_input, model
        return ["decoded-single"], 3, provided_token_lengths

    with pytest.raises(EmbeddingInputError) as exc_info:
        normalize_embedding_input(
            [101, 102, 103],
            model="test-model",
            max_tokens=100,
            count_tokens=_count_tokens,
            tokens_to_texts=malformed_tokens_to_texts,
        )

    _assert_input_error(exc_info, "invalid_token_array", "Invalid token array input")


@pytest.mark.unit
def test_token_array_decoder_absent_token_lengths_falls_back_to_raw_lengths():
    def two_value_tokens_to_texts(tokens_input, model):
        del tokens_input, model
        return ["decoded-0", "decoded-1"], 5

    normalized = normalize_embedding_input(
        [[101, 102], [201, 202, 203]],
        model="test-model",
        max_tokens=100,
        count_tokens=_count_tokens,
        tokens_to_texts=two_value_tokens_to_texts,
    )

    assert normalized.texts == ["decoded-0", "decoded-1"]
    assert normalized.token_counts == [2, 3]
    assert normalized.total_tokens == 5
    assert normalized.provided_token_arrays is True
    assert normalized.token_input_mode == "batch"


@pytest.mark.unit
def test_single_token_array_decoder_length_mismatch_rejected_even_when_numeric():
    def mismatched_tokens_to_texts(tokens_input, model):
        del tokens_input, model
        return ["decoded-single"], 1, [1]

    with pytest.raises(EmbeddingInputError) as exc_info:
        normalize_embedding_input(
            [101, 102, 103],
            model="test-model",
            max_tokens=2,
            count_tokens=_count_tokens,
            tokens_to_texts=mismatched_tokens_to_texts,
        )

    _assert_input_error(exc_info, "invalid_token_array", "Invalid token array input")


@pytest.mark.unit
def test_batch_token_array_decoder_length_mismatch_rejected_even_when_numeric():
    def mismatched_tokens_to_texts(tokens_input, model):
        del tokens_input, model
        return ["decoded-0", "decoded-1"], 3, [1, 1]

    with pytest.raises(EmbeddingInputError) as exc_info:
        normalize_embedding_input(
            [[101], [201, 202]],
            model="test-model",
            max_tokens=1,
            count_tokens=_count_tokens,
            tokens_to_texts=mismatched_tokens_to_texts,
        )

    _assert_input_error(exc_info, "invalid_token_array", "Invalid token array input")


@pytest.mark.unit
@pytest.mark.parametrize(
    "decoded_texts",
    [
        "x",
        [123],
    ],
)
def test_token_array_decoder_malformed_decoded_texts_maps_to_invalid_token_array(decoded_texts):
    def malformed_tokens_to_texts(tokens_input, model):
        del tokens_input, model
        return decoded_texts, 3, [3]

    with pytest.raises(EmbeddingInputError) as exc_info:
        normalize_embedding_input(
            [101, 102, 103],
            model="test-model",
            max_tokens=100,
            count_tokens=_count_tokens,
            tokens_to_texts=malformed_tokens_to_texts,
        )

    _assert_input_error(exc_info, "invalid_token_array", "Invalid token array input")


@pytest.mark.unit
def test_token_limit_for_text_input_reports_offending_index_and_count():
    def count_tokens(text: str, model: str) -> int:
        del model
        return len(text)

    with pytest.raises(EmbeddingInputError) as exc_info:
        normalize_embedding_input(
            "abcdef",
            model="test-model",
            max_tokens=5,
            count_tokens=count_tokens,
            tokens_to_texts=_tokens_to_texts,
        )

    _assert_input_error(
        exc_info,
        "input_too_long",
        "One or more inputs exceed max tokens 5 for model test-model",
        details=[{"index": 0, "tokens": 6}],
    )


@pytest.mark.unit
def test_token_limit_for_token_array_batch_reports_offending_index_and_raw_length():
    with pytest.raises(EmbeddingInputError) as exc_info:
        _normalize([[101], [201, 202, 203]], max_tokens=2)

    _assert_input_error(
        exc_info,
        "input_too_long",
        "One or more inputs exceed max tokens 2 for model test-model",
        details=[{"index": 1, "tokens": 3}],
    )
