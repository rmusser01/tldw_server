"""Deterministic alias and namespace rules for study suggestion topics."""

from __future__ import annotations

from collections.abc import Iterable
import re

NORMALIZATION_VERSION = "norm-v2"
DEFAULT_NAMESPACE = "general"

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_TOKEN_ALIASES: dict[str, str] = {
    "kidney": "renal",
    "kidneys": "renal",
    "nephrology": "renal",
    "heart": "cardiac",
    "hearts": "cardiac",
    "cardiology": "cardiac",
    "brain": "neuro",
    "brains": "neuro",
    "neural": "neuro",
    "neuronal": "neuro",
}

NAMESPACE_RULES: dict[str, tuple[str, ...]] = {
    "renal": ("renal", "kidney", "kidneys", "nephrology"),
    "cardiac": ("cardiac", "heart", "hearts", "cardiology"),
    "neuro": ("neuro", "brain", "brains", "neural", "neuronal"),
}

EXACT_TOPIC_ALIASES: dict[str, tuple[str, str]] = {
    "kidney physiology": ("renal", "renal physiology"),
    "renal physiology": ("renal", "renal physiology"),
    "renal-physiology": ("renal", "renal physiology"),
    "kidney basics": ("renal", "renal basics"),
    "renal basics": ("renal", "renal basics"),
    "renal overview": ("renal", "overview"),
    "cardiac overview": ("cardiac", "overview"),
    "cardiac physiology": ("cardiac", "cardiac physiology"),
    "heart physiology": ("cardiac", "cardiac physiology"),
}


def clean_label_text(label: object) -> str | None:
    text = str(label or "").strip().lower()
    if not text:
        return None
    text = re.sub(r"[-/_]", " ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip() or None


def normalize_semantic_label(label: object) -> str | None:
    cleaned = clean_label_text(label)
    if cleaned is None:
        return None
    tokens = [_TOKEN_ALIASES.get(token, token) for token in cleaned.split()]
    return " ".join(tokens).strip() or None


def resolve_namespace(label: str) -> str:
    tokens = label.split()
    for namespace, aliases in NAMESPACE_RULES.items():
        if any(alias in tokens for alias in aliases):
            return namespace
    return DEFAULT_NAMESPACE


def lookup_topic_alias(label: str) -> tuple[str, str] | None:
    alias = EXACT_TOPIC_ALIASES.get(label)
    if alias is not None:
        return alias
    return None


def resolve_topic_alias(label: str) -> tuple[str, str]:
    cleaned = clean_label_text(label)
    if cleaned is None:
        return DEFAULT_NAMESPACE, ""
    semantic = normalize_semantic_label(cleaned) or cleaned
    alias = lookup_topic_alias(cleaned) or lookup_topic_alias(semantic)
    if alias is not None:
        return alias
    return resolve_namespace(semantic), semantic


def canonical_slug(label: str) -> str:
    slug = _NON_ALNUM_RE.sub("-", label.lower()).strip("-")
    slug = _WHITESPACE_RE.sub("-", slug)
    return slug or DEFAULT_NAMESPACE


def topic_key_for(namespace: str, canonical_label: str) -> str:
    return f"{namespace}:{canonical_slug(canonical_label)}"


def dedupe_reasons(reasons: Iterable[str]) -> list[str]:
    unique: list[str] = []
    for reason in reasons:
        if reason and reason not in unique:
            unique.append(reason)
    return unique
