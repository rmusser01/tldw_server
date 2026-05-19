"""Load shared deterministic Persona Chat quality fixture cases for tests."""

from __future__ import annotations

from copy import deepcopy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "persona_chat_quality_cases.json"


@lru_cache(maxsize=1)
def _cached_cases() -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


def all_cases() -> tuple[dict[str, Any], ...]:
    """Return independent copies of all Persona Chat quality fixture cases."""
    return tuple(deepcopy(case) for case in _cached_cases())


def case_by_id(case_id: str) -> dict[str, Any]:
    """Return an independent copy of the fixture case matching ``case_id``."""
    for case in _cached_cases():
        if case.get("case_id") == case_id:
            return deepcopy(case)
    raise KeyError(f"Unknown Persona Chat quality fixture case: {case_id}")
