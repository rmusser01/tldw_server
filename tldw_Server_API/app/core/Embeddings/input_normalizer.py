"""Pure input normalization for embedding requests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingDomainError,
    EmbeddingErrorCode,
    EmbeddingInputError,
    NormalizedEmbeddingInput,
)

_MAX_INPUTS = 2048
TokenInputMode = Literal["none", "single", "batch"]


def _input_error(
    code: EmbeddingErrorCode,
    message: str,
    *,
    details: list[dict[str, int]] | None = None,
) -> EmbeddingInputError:
    return EmbeddingInputError(code, message, details=details)


def _is_token_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _raise_if_too_long(token_counts: list[int], *, max_tokens: int, model: str) -> None:
    too_long = [
        {"index": index, "tokens": tokens}
        for index, tokens in enumerate(token_counts)
        if tokens > max_tokens
    ]
    if too_long:
        raise _input_error(
            "input_too_long",
            f"One or more inputs exceed max tokens {max_tokens} for model {model}",
            details=too_long,
        )


def _normalize_texts(
    texts: list[str],
    *,
    model: str,
    max_tokens: int,
    count_tokens: Callable[[str, str], int],
    empty_message: str = "Input cannot be empty",
) -> NormalizedEmbeddingInput:
    for text in texts:
        if not text.strip():
            raise _input_error("empty_input", empty_message)

    token_counts = [int(count_tokens(text, model) or 0) for text in texts]
    _raise_if_too_long(token_counts, max_tokens=max_tokens, model=model)
    return NormalizedEmbeddingInput(texts, token_counts, sum(token_counts), False, "none")


def _raw_token_lengths(raw_input: list[int] | list[list[int]], mode: TokenInputMode) -> list[int]:
    if mode == "single":
        if not all(_is_token_id(token) for token in raw_input):
            raise _input_error("invalid_token_array", "Invalid token array input")
        return [len(raw_input)]

    token_batches = raw_input
    if not all(isinstance(item, list) and all(_is_token_id(token) for token in item) for item in token_batches):
        raise _input_error("invalid_token_array", "Invalid token array input")
    return [len(item) for item in token_batches]


def _validate_provided_token_lengths(value: object, expected_lengths: list[int]) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _input_error("invalid_token_array", "Invalid token array input")

    token_lengths: list[int] = []
    for item in value:
        if not _is_token_id(item) or item < 0:
            raise _input_error("invalid_token_array", "Invalid token array input")
        token_lengths.append(item)
    if token_lengths != expected_lengths:
        raise _input_error("invalid_token_array", "Invalid token array input")
    return token_lengths


def _validate_decoded_texts(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _input_error("invalid_token_array", "Invalid token array input")

    texts: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise _input_error("invalid_token_array", "Invalid token array input")
        texts.append(item)
    return texts


def _raise_if_token_array_shape_mismatch(texts: list[str], token_lengths: list[int], expected_count: int) -> None:
    if len(texts) != expected_count or len(token_lengths) != expected_count:
        raise _input_error("invalid_token_array", "Invalid token array input")


def _normalize_token_arrays(
    raw_input: list[int] | list[list[int]],
    *,
    model: str,
    max_tokens: int,
    mode: Literal["single", "batch"],
    tokens_to_texts: Callable[[list[int] | list[list[int]], str], object],
) -> NormalizedEmbeddingInput:
    fallback_lengths = _raw_token_lengths(raw_input, mode)
    expected_count = 1 if mode == "single" else len(raw_input)
    try:
        decoded = tokens_to_texts(raw_input, model)
        texts = _validate_decoded_texts(decoded[0])  # type: ignore[index]
        token_lengths = (
            _validate_provided_token_lengths(decoded[2], fallback_lengths)
            if len(decoded) > 2
            else fallback_lengths
        )  # type: ignore[arg-type]
    except EmbeddingDomainError:
        raise
    except Exception as exc:
        raise _input_error("invalid_token_array", "Invalid token array input") from exc

    _raise_if_token_array_shape_mismatch(texts, token_lengths, expected_count)
    _raise_if_too_long(token_lengths, max_tokens=max_tokens, model=model)
    return NormalizedEmbeddingInput(
        texts=texts,
        token_counts=token_lengths,
        total_tokens=sum(token_lengths),
        provided_token_arrays=True,
        token_input_mode=mode,
    )


def normalize_embedding_input(
    raw_input: Any,
    *,
    model: str,
    max_tokens: int,
    count_tokens: Callable[[str, str], int],
    tokens_to_texts: Callable[[list[int] | list[list[int]], str], object],
) -> NormalizedEmbeddingInput:
    """Normalize supported embedding input shapes into texts and token counts."""
    if isinstance(raw_input, str):
        return _normalize_texts([raw_input], model=model, max_tokens=max_tokens, count_tokens=count_tokens)

    if not isinstance(raw_input, list):
        raise _input_error("invalid_input_type", "Invalid input type")

    if not raw_input:
        raise _input_error("empty_input", "Input list cannot be empty")

    if all(isinstance(item, str) for item in raw_input):
        if len(raw_input) > _MAX_INPUTS:
            raise _input_error("too_many_inputs", "Maximum 2048 inputs allowed")
        return _normalize_texts(
            raw_input,
            model=model,
            max_tokens=max_tokens,
            count_tokens=count_tokens,
            empty_message="Input list cannot contain empty strings",
        )

    if all(_is_token_id(item) for item in raw_input):
        return _normalize_token_arrays(
            raw_input,
            model=model,
            max_tokens=max_tokens,
            mode="single",
            tokens_to_texts=tokens_to_texts,
        )

    if all(isinstance(item, list) for item in raw_input):
        if len(raw_input) > _MAX_INPUTS:
            raise _input_error("too_many_inputs", "Maximum 2048 inputs allowed")
        return _normalize_token_arrays(
            raw_input,
            model=model,
            max_tokens=max_tokens,
            mode="batch",
            tokens_to_texts=tokens_to_texts,
        )

    raise _input_error("invalid_input_type", "Invalid input type")


__all__ = ["normalize_embedding_input"]
