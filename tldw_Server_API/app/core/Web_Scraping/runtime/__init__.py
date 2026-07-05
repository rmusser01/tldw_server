"""Runtime contracts and adapters for the staged Web_Scraping refactor."""

from __future__ import annotations

from .requests import FetchRequest, RuntimeRequestContext
from .responses import FetchResponse, PolicyDecision

__all__ = [
    "FetchRequest",
    "FetchResponse",
    "PolicyDecision",
    "RuntimeRequestContext",
]
