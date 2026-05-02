"""
Synonyms/Alias Registry per corpus/namespace.

Loads optional JSON files from Config_Files/Synonyms/<corpus>.json with a mapping of
term -> [aliases]. Used to enrich both FTS (where integrated) and query rewrites.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger


def _find_project_root(start: Path | None = None) -> Path:
    """Locate the project root by finding a parent that contains 'tldw_Server_API'."""
    p = (start or Path(__file__).resolve())
    for anc in p.parents:
        if (anc / "tldw_Server_API").exists():
            return anc
    return p.parents[-1]


def _get_config_root() -> Path:
    """Return the directory that holds the primary configuration files."""
    # Highest priority: explicit environment variables
    env_config_path = os.getenv("TLDW_CONFIG_PATH")
    if env_config_path:
        cfg_path = Path(env_config_path).expanduser()
        if cfg_path.is_file():
            root = cfg_path.parent
            logger.debug(f"Synonyms config root via TLDW_CONFIG_PATH file parent: {root}")
            return root
        if cfg_path.is_dir():
            logger.debug(f"Synonyms config root via TLDW_CONFIG_PATH dir: {cfg_path}")
            return cfg_path

    env_config_dir = os.getenv("TLDW_CONFIG_DIR")
    if env_config_dir:
        cfg_dir = Path(env_config_dir).expanduser()
        if cfg_dir.exists():
            logger.debug(f"Synonyms config root via TLDW_CONFIG_DIR: {cfg_dir}")
            return cfg_dir

    # Fallback to repo defaults
    base = _find_project_root()
    candidates = [
        base / "tldw_Server_API" / "Config_Files",
        base / "Config_Files",
    ]
    for candidate in candidates:
        if candidate.exists():
            logger.debug(f"Synonyms config root via repo candidate: {candidate}")
            return candidate

    # Return the first candidate even if it does not currently exist so callers
    # have a deterministic location to create.
    logger.debug(f"Synonyms config root default (non-existent): {candidates[0]}")
    return candidates[0]


def get_corpus_synonyms(corpus: str | None) -> dict[str, list[str]]:
    if not corpus:
        return {}

    config_root = _get_config_root()

    candidates: list[Path] = [
        config_root / "Synonyms" / f"{corpus}.json",
        config_root / f"{corpus}.json",
    ]

    # Back-compatibility: honour historically searched locations in case files
    # are still mounted there (e.g., legacy deployments).
    base = _find_project_root()
    candidates.extend([
        base / "tldw_Server_API" / "Config_Files" / "Synonyms" / f"{corpus}.json",
        base / "Config_Files" / "Synonyms" / f"{corpus}.json",
    ])

    start = Path(__file__).resolve()
    for anc in start.parents:
        candidates.append(anc / "tldw_Server_API" / "Config_Files" / "Synonyms" / f"{corpus}.json")
        candidates.append(anc / "Config_Files" / "Synonyms" / f"{corpus}.json")

    # Deduplicate while preserving order
    seen: set[Path] = set()
    ordered_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            ordered_candidates.append(candidate)
            seen.add(candidate)

    path = next((c for c in ordered_candidates if c.exists()), ordered_candidates[0])
    logger.debug(
        "Synonyms file selection: path='{}' exists={} corpus='{}'",
        path,
        path.exists(),
        corpus,
    )
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # normalize to str -> list[str]
                    out: dict[str, list[str]] = {}
                    for k, v in data.items():
                        if isinstance(k, str):
                            if isinstance(v, list):
                                out[k.lower()] = [str(x).lower() for x in v]
                            elif isinstance(v, str):
                                out[k.lower()] = [v.lower()]
                    return out
    except Exception:
        logger.warning(f"Failed to load synonyms for corpus '{corpus}'")
    return {}
