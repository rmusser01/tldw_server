"""Exact source partitions for local embedding token limits."""

from collections.abc import Callable
from typing import Any


def effective_token_limit(configured: int, tokenizer: Any) -> int:
    """Use the finite configured cap and any stricter tokenizer model cap."""
    if isinstance(configured, bool) or not isinstance(configured, int) or configured < 1:
        raise ValueError("Embedding token budget must be a positive integer")
    model_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_limit, int) and not isinstance(model_limit, bool) and 0 < model_limit < 1_000_000_000:
        return min(configured, model_limit)
    return configured


def split_chunks_to_token_budget(
    chunks: list[dict[str, Any]],
    tokenizer: Any,
    max_tokens: int,
    *,
    format_text: Callable[[str], str] = lambda text: text,
) -> list[dict[str, Any]]:
    """Partition exact strings; tokenization only measures, never rewrites source text."""
    limit = effective_token_limit(max_tokens, tokenizer)
    output = []
    for chunk in chunks:
        text, start, end = chunk.get("text"), chunk.get("start"), chunk.get("end")
        if (
            not isinstance(text, str)
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end - start != len(text)
        ):
            raise ValueError("Embedding chunk has invalid canonical offsets")
        pending = [(0, len(text))]
        while pending:
            left, right = pending.pop()
            part = text[left:right]
            count = len(tokenizer.encode(format_text(part), add_special_tokens=True, truncation=False))
            if count <= limit:
                output.append(
                    {
                        **chunk,
                        "text": part,
                        "start": start + left,
                        "end": start + right,
                        "index": len(output),
                        "metadata": {
                            **(chunk.get("metadata") or {}),
                            "embedding_token_count": count,
                            "embedding_max_tokens": limit,
                        },
                    }
                )
                continue
            if right - left <= 1:
                raise ValueError("A source character and model formatting exceed the embedding token budget")
            midpoint = left + (right - left) // 2
            # Prefer a nearby word boundary, without dropping whitespace or
            # relying on token offsets/decoded strings that may normalize Unicode.
            boundary = text.rfind(" ", left + (right - left) // 4, midpoint + 1)
            if left < boundary + 1 < right:
                midpoint = boundary + 1
            pending.extend(((midpoint, right), (left, midpoint)))
    return output


def validate_token_budget(texts: list[str], tokenizer: Any, max_tokens: int) -> None:
    """Reject overlong raw inputs before padded tensor allocation or provider work."""
    from tldw_Server_API.app.core.Embeddings.request_types import EmbeddingInputError

    limit = effective_token_limit(max_tokens, tokenizer)
    too_long = []
    for index, text in enumerate(texts):
        count = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        if count > limit:
            too_long.append({"index": index, "tokens": count})
    if too_long:
        raise EmbeddingInputError(
            "input_too_long",
            f"Embedding inputs exceed the model token budget ({limit}); split them before embedding",
            details=too_long,
        )
