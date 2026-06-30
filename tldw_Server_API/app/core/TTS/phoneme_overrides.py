"""
phoneme_overrides.py

Utility functions for loading and applying phoneme/lexicon overrides.

Supports YAML or JSON configuration files and request/provider-level overrides.
Entries are applied on word boundaries by default to avoid partial matches.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


# Cap the number of override entries to avoid unbounded processing.
_MAX_OVERRIDE_ENTRIES = 256


@dataclass
class PhonemeOverrideEntry:
    """
    Represents a single phoneme override rule.

    Attributes:
        term: The word or phrase to match in the input text.
        phonemes: The phonetic representation to substitute (e.g., IPA notation).
        lang: Optional language code to scope the override (e.g., "en", "en-US").
        boundary: If True (default), match only on word boundaries; if False, allow mid-word matches.
        provider: Optional TTS provider name to scope this override (e.g., "kokoro", "elevenlabs").
    """

    term: str
    phonemes: str
    lang: str | None = None
    boundary: bool = True
    provider: str | None = None


def _resolve_config_path(path_hint: str | None) -> Path | None:
    """Resolve an optional path hint to an existing config file."""
    candidates: list[str] = []
    # Explicit override (arg, env)
    if path_hint:
        candidates.append(path_hint)
    env_path = os.getenv("TTS_PHONEME_OVERRIDES_PATH")
    if env_path:
        candidates.append(env_path)

    # Default locations under Config_Files
    base_dir = Path(__file__).resolve().parents[3] / "Config_Files"
    for fname in ("tts_phonemes.yaml", "tts_phonemes.yml", "tts_phonemes.json"):
        candidates.append(str(base_dir / fname))

    for cand in candidates:
        try:
            p = Path(cand)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if p.exists() and p.is_file():
                return p
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Skipping invalid config path candidate "
                f"(error_type={exc.__class__.__name__})"
            )
            continue
    return None


def _coerce_entry(raw: Any, provider_hint: str | None) -> PhonemeOverrideEntry | None:
    """Coerce a raw object into a PhonemeOverrideEntry."""
    try:
        if isinstance(raw, PhonemeOverrideEntry):
            return raw
        if isinstance(raw, dict):
            term = raw.get("term") or raw.get("word") or raw.get("token")
            phonemes = raw.get("phonemes") or raw.get("phoneme") or raw.get("ipa")
            if not term or not phonemes:
                return None
            return PhonemeOverrideEntry(
                term=str(term).strip(),
                phonemes=str(phonemes).strip(),
                lang=(raw.get("lang") or raw.get("language")),
                boundary=bool(raw.get("boundary", True)),
                provider=raw.get("provider") or provider_hint,
            )
        if isinstance(raw, tuple) and len(raw) >= 2:
            term, phonemes = raw[0], raw[1]
            return PhonemeOverrideEntry(term=str(term), phonemes=str(phonemes), provider=provider_hint)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "Failed to parse phoneme override entry "
            f"(error_type={exc.__class__.__name__})"
        )
    return None


def parse_override_entries(raw: Any, provider_hint: str | None = None) -> list[PhonemeOverrideEntry]:
    """
    Parse user-supplied overrides into normalized entries.

    Accepts:
      - list[dict] with term/phonemes keys
      - dict mapping term -> phoneme string
      - list of (term, phoneme) tuples
    """
    entries: list[PhonemeOverrideEntry] = []
    if raw is None:
        return entries

    # Structured config shape:
    # {
    #   "global": [...],
    #   "providers": {"kokoro": [...], "other": [...]}
    # }
    if isinstance(raw, dict) and ("global" in raw or "providers" in raw):
        global_entries = parse_override_entries(raw.get("global"), provider_hint=None)
        entries.extend(global_entries)

        providers_raw = raw.get("providers")
        if isinstance(providers_raw, dict):
            for provider_name, provider_entries_raw in providers_raw.items():
                provider_key = str(provider_name).strip().lower()
                if not provider_key:
                    continue
                provider_entries = parse_override_entries(provider_entries_raw, provider_hint=provider_key)
                entries.extend(provider_entries)
        return _normalize_entries(entries)

    # Dict mapping term -> phoneme
    if isinstance(raw, dict) and not {"term", "phonemes"} <= set(raw.keys()):
        for term, phonemes in raw.items():
            # Keep mapping mode strict: values should be scalars.
            if isinstance(phonemes, (dict, list, tuple)):
                logger.debug(f"Skipping invalid phoneme mapping for '{term}': expected scalar value")
                continue
            ent = _coerce_entry({"term": term, "phonemes": phonemes}, provider_hint)
            if ent:
                entries.append(ent)
        return _normalize_entries(entries)

    # List/tuple payloads or single dict
    if isinstance(raw, (list, tuple)):
        for item in raw:
            ent = _coerce_entry(item, provider_hint)
            if ent:
                entries.append(ent)
    else:
        ent = _coerce_entry(raw, provider_hint)
        if ent:
            entries.append(ent)

    return _normalize_entries(entries)


def _normalize_entries(entries: Sequence[PhonemeOverrideEntry]) -> list[PhonemeOverrideEntry]:
    """Normalize terms/providers and clamp list size."""
    cleaned: list[PhonemeOverrideEntry] = []
    for ent in entries:
        term = str(ent.term or "").strip()
        phonemes = str(ent.phonemes or "").strip()
        if not term or not phonemes:
            continue
        lang = str(ent.lang).strip() if ent.lang is not None else None
        provider = str(ent.provider).strip().lower() if ent.provider is not None else None
        cleaned.append(
            PhonemeOverrideEntry(
                term=term,
                phonemes=phonemes,
                lang=(lang or None),
                boundary=bool(ent.boundary),
                provider=(provider or None),
            )
        )
    return cleaned[:_MAX_OVERRIDE_ENTRIES]


def load_override_entries(path_hint: str | None = None) -> list[PhonemeOverrideEntry]:
    """
    Load phoneme overrides from YAML or JSON.

    Cache is keyed by the resolved path so repeated calls are cheap, even
    when different hints resolve to the same underlying file.
    """
    path = _resolve_config_path(path_hint)
    if not path:
        return []

    return _load_override_entries_cached(str(path))


@lru_cache(maxsize=4)
def _load_override_entries_cached(path_str: str) -> list[PhonemeOverrideEntry]:
    """Internal cached loader keyed by resolved path string."""
    path = Path(path_str)

    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                logger.warning("PyYAML not installed; skipping phoneme override load")
                return []
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception as exc:
        logger.warning(
            "Failed to load phoneme overrides "
            f"(error_type={exc.__class__.__name__})"
        )
        return []

    entries = parse_override_entries(data)
    if len(entries) > _MAX_OVERRIDE_ENTRIES:
        logger.warning(
            f"Phoneme override file {path} has {len(entries)} entries; "
            f"capping to {_MAX_OVERRIDE_ENTRIES} to avoid performance regressions"
        )
        entries = entries[:_MAX_OVERRIDE_ENTRIES]
    return entries


def filter_overrides_for_provider(
    entries: Sequence[PhonemeOverrideEntry],
    provider: str | None,
) -> list[PhonemeOverrideEntry]:
    """Return entries applicable to a given provider (None or case-insensitive match)."""
    if not provider:
        return list(entries)
    pl = provider.lower()
    return [e for e in entries if not e.provider or str(e.provider).lower() == pl]


def merge_override_entries(*entry_sets: Iterable[PhonemeOverrideEntry]) -> list[PhonemeOverrideEntry]:
    """
    Merge multiple override sources with later sets taking precedence.

    Keyed by (term_lower, lang_lower) to allow per-language variants.
    """
    merged: dict[tuple[str, str], PhonemeOverrideEntry] = {}
    for entries in entry_sets:
        for ent in entries:
            if not ent.term or not ent.phonemes:
                continue
            key = (ent.term.lower(), (ent.lang or "").lower())
            merged[key] = ent
    return _normalize_entries(list(merged.values()))


def apply_overrides_to_text(
    text: str,
    entries: Sequence[PhonemeOverrideEntry],
    *,
    lang_hint: str | None = None,
) -> str:
    """
    Apply override entries to text by replacing matches with [[phoneme]] tokens.

    - Respects lang_hint when entry.lang is set (prefix match, case-insensitive).
    - Uses word-boundary matching by default; set entry.boundary=False to allow mid-word.
    """
    if not entries or not text:
        return text

    lang_prefix = (lang_hint or "").split("-")[0].lower() if lang_hint else None
    updated = text

    # Resolve overlap deterministically: apply longer terms first.
    sorted_entries = sorted(entries, key=lambda e: len(e.term), reverse=True)
    for ent in sorted_entries:
        if ent.lang and lang_prefix and not ent.lang.lower().startswith(lang_prefix):
            continue
        pattern = re.compile(
            rf"\b{re.escape(ent.term)}\b" if ent.boundary else re.escape(ent.term),
            flags=re.IGNORECASE,
        )

        # Preserve surrounding whitespace/punctuation by replacing only the term
        def _repl(_: re.Match[str], phonemes: str = ent.phonemes) -> str:
            return f"[[{phonemes}]]"

        updated = pattern.sub(_repl, updated)
    return updated
