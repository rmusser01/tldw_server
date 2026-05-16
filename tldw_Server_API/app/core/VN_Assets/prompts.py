"""Prompt preview helpers for VN asset generation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from math import ceil
from typing import Any

try:
    from requests import RequestException
except ImportError:  # pragma: no cover - requests is an application dependency.
    RequestException = None  # type: ignore[assignment]


DEFAULT_CHARACTER_BUDGET = 1500
DEFAULT_WORLD_BOOK_BUDGET = 1000
DEFAULT_PACK_BUDGET = 750
DEFAULT_SLOT_BUDGET = 750
DEFAULT_TOTAL_BUDGET = 4000

SOURCE_BUCKETS = ("slot", "pack", "character", "world_book", "negative_prompt")
TOKEN_ENCODER_FALLBACK_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
if RequestException is not None:
    TOKEN_ENCODER_FALLBACK_EXCEPTIONS = TOKEN_ENCODER_FALLBACK_EXCEPTIONS + (RequestException,)


@dataclass(frozen=True, slots=True)
class PromptBudgets:
    """Estimated-token budgets for prompt source buckets."""

    character: int = DEFAULT_CHARACTER_BUDGET
    world_book: int = DEFAULT_WORLD_BOOK_BUDGET
    pack: int = DEFAULT_PACK_BUDGET
    slot: int = DEFAULT_SLOT_BUDGET
    total: int = DEFAULT_TOTAL_BUDGET

    def __post_init__(self) -> None:
        for field_name in ("character", "world_book", "pack", "slot", "total"):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name}_budget_must_be_non_negative")


@dataclass(slots=True)
class PromptPreview:
    """Assembled prompt preview plus source-budget diagnostics.

    ``omitted_source_counts`` stores omitted estimated-token counts per bucket.
    The field name follows the public Task 3 contract.
    """

    prompt: str
    negative_prompt: str = ""
    omitted_source_counts: dict[str, int] = field(default_factory=dict)
    token_estimates: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        omitted = {bucket: 0 for bucket in SOURCE_BUCKETS}
        omitted.update(self.omitted_source_counts)
        self.omitted_source_counts = omitted
        self.warnings = tuple(self.warnings)


def build_prompt_preview(
    *,
    character: Any,
    world_book_entries: Iterable[Any] | None = None,
    pack_style: str | None = None,
    slot_template: str | None = None,
    labels: Mapping[str, Any] | None = None,
    budgets: PromptBudgets | None = None,
    negative_prompt: str | None = None,
    pack_scenario: str | None = None,
    style_lock: Mapping[str, Any] | None = None,
) -> PromptPreview:
    """Build a deterministic prompt preview with source budget enforcement."""
    effective_budgets = budgets or PromptBudgets()
    omitted_source_counts = {bucket: 0 for bucket in SOURCE_BUCKETS}
    token_estimates: dict[str, int] = {}
    warnings: list[str] = []

    assembled_parts: list[str] = []

    negative_source = _normalize_text(negative_prompt)

    for bucket, raw_text, bucket_budget in (
        ("slot", _build_slot_source(slot_template, labels), effective_budgets.slot),
        (
            "pack",
            _build_pack_source(
                pack_scenario=pack_scenario,
                pack_style=pack_style,
                style_lock=style_lock,
            ),
            effective_budgets.pack,
        ),
        ("character", _build_character_source(character), effective_budgets.character),
        (
            "world_book",
            _build_world_book_source(world_book_entries or ()),
            effective_budgets.world_book,
        ),
    ):
        if not raw_text:
            token_estimates[bucket] = 0
            continue

        truncated_text, omitted = _truncate_to_join_budget(
            raw_text,
            bucket_budget,
            existing_parts=assembled_parts,
            total_budget=effective_budgets.total,
        )
        if truncated_text:
            assembled_parts.append(truncated_text)
        used = estimate_prompt_tokens(truncated_text)
        token_estimates[bucket] = used
        omitted_source_counts[bucket] = omitted
        if omitted > 0:
            warnings.append(f"{bucket} truncated by {omitted} estimated tokens")

    negative_text, negative_omitted = _truncate_to_budget(negative_source, effective_budgets.pack)
    token_estimates["negative_prompt"] = estimate_prompt_tokens(negative_text)
    omitted_source_counts["negative_prompt"] = negative_omitted
    if negative_omitted > 0:
        warnings.append(f"negative_prompt truncated by {negative_omitted} estimated tokens")

    prompt = "\n\n".join(assembled_parts)
    token_estimates["total"] = estimate_prompt_tokens(prompt)

    return PromptPreview(
        prompt=prompt,
        negative_prompt=negative_text,
        omitted_source_counts=omitted_source_counts,
        token_estimates=token_estimates,
        warnings=tuple(warnings),
    )


def estimate_prompt_tokens(text: str | None) -> int:
    """Estimate tokens using a tokenizer when available, with a deterministic fallback."""
    normalized = _normalize_text(text)
    if not normalized:
        return 0
    encoder = _get_prompt_token_encoder()
    if encoder is not None:
        try:
            return max(0, len(encoder.encode(normalized)))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    word_count = len(normalized.split())
    char_count = len(normalized)
    return max(word_count, ceil(char_count / 4))


@lru_cache(maxsize=1)
def _get_prompt_token_encoder() -> Any | None:
    try:
        import tiktoken  # type: ignore
    except ImportError:
        return None

    try:
        return tiktoken.get_encoding("cl100k_base")
    except TOKEN_ENCODER_FALLBACK_EXCEPTIONS:
        return None


def _truncate_to_budget(text: str, budget: int) -> tuple[str, int]:
    normalized = _normalize_text(text)
    original_tokens = estimate_prompt_tokens(normalized)
    if original_tokens <= budget:
        return normalized, 0
    if budget <= 0:
        return "", original_tokens

    words = normalized.split()
    low = 0
    high = len(words)
    best = ""
    while low <= high:
        midpoint = (low + high) // 2
        candidate = " ".join(words[:midpoint])
        if estimate_prompt_tokens(candidate) <= budget:
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1

    if not best:
        best = normalized[: budget * 4].strip()
        while best and estimate_prompt_tokens(best) > budget:
            best = best[:-1].strip()

    omitted = max(original_tokens - estimate_prompt_tokens(best), 1)
    return best, omitted


def _truncate_to_join_budget(
    text: str,
    bucket_budget: int,
    *,
    existing_parts: list[str],
    total_budget: int,
) -> tuple[str, int]:
    truncated, omitted = _truncate_to_budget(text, bucket_budget)
    while truncated and estimate_prompt_tokens("\n\n".join(existing_parts + [truncated])) > total_budget:
        reduced, _ = _truncate_to_budget(truncated, max(estimate_prompt_tokens(truncated) - 1, 0))
        if reduced == truncated:
            truncated = ""
        else:
            truncated = reduced
    if not truncated:
        original_tokens = estimate_prompt_tokens(text)
        return "", max(original_tokens, omitted)
    original_tokens = estimate_prompt_tokens(text)
    return truncated, max(original_tokens - estimate_prompt_tokens(truncated), omitted)


def _build_slot_source(
    slot_template: str | None,
    labels: Mapping[str, Any] | None,
) -> str:
    parts: list[str] = []
    template = _normalize_text(slot_template)
    if template:
        parts.append(f"Slot template: {template}")
    if labels:
        label_text = ", ".join(
            f"{key}={_stringify_value(labels[key])}" for key in sorted(labels, key=lambda value: str(value))
        )
        if label_text:
            parts.append(f"Labels: {label_text}")
    return "\n".join(parts)


def _build_pack_source(
    *,
    pack_scenario: str | None,
    pack_style: str | None,
    style_lock: Mapping[str, Any] | None,
) -> str:
    parts: list[str] = []
    scenario = _normalize_text(pack_scenario)
    style = _normalize_text(pack_style)
    if style:
        parts.append(f"Pack style: {style}")
    if style_lock:
        style_lock_text = ", ".join(
            f"{key}={_stringify_value(style_lock[key])}" for key in sorted(style_lock, key=lambda value: str(value))
        )
        if style_lock_text:
            parts.append(f"Style lock: {style_lock_text}")
    if scenario:
        parts.append(f"Pack scenario: {scenario}")
    return "\n".join(parts)


def _build_character_source(character: Any) -> str:
    parts: list[str] = []
    for field_name, label in (
        ("name", "Character name"),
        ("description", "Description"),
        ("personality", "Personality"),
        ("scenario", "Character scenario"),
        ("first_message", "First message"),
        ("creator_notes", "Creator notes"),
        ("style_notes", "Style notes"),
        ("image_notes", "Image notes"),
        ("image_style", "Image style"),
        ("avatar_style", "Avatar style"),
        ("appearance", "Appearance"),
    ):
        value = _get_field(character, field_name)
        text = _normalize_text(value)
        if text:
            parts.append(f"{label}: {text}")
    return "\n".join(parts)


def _build_world_book_source(world_book_entries: Iterable[Any]) -> str:
    normalized_entries: list[tuple[int, int, str]] = []
    for index, entry in enumerate(world_book_entries):
        text = _world_book_text(entry)
        if not text:
            continue
        priority = _world_book_priority(entry)
        normalized_entries.append((priority, index, text))

    parts = [
        f"World book: {text}"
        for priority, index, text in sorted(
            normalized_entries,
            key=lambda item: (-item[0], item[1]),
        )
    ]
    return "\n".join(parts)


def _world_book_text(entry: Any) -> str:
    if isinstance(entry, str):
        return _normalize_text(entry)
    for field_name in ("text", "content", "summary"):
        text = _normalize_text(_get_field(entry, field_name))
        if text:
            return text
    return _normalize_text(entry)


def _world_book_priority(entry: Any) -> int:
    value = _get_field(entry, "priority")
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _get_field(source: Any, field_name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(field_name)
    return getattr(source, field_name, None)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return " ".join(str(value).split())


def _stringify_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return (
            "{"
            + ", ".join(f"{key}: {_stringify_value(value[key])}" for key in sorted(value, key=lambda item: str(item)))
            + "}"
        )
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_stringify_value(item) for item in value) + "]"
    return _normalize_text(value)


__all__ = [
    "PromptBudgets",
    "PromptPreview",
    "build_prompt_preview",
    "estimate_prompt_tokens",
]
