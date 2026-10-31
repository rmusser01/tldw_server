"""Policy-bound target contract for governed preflight analysis."""

from __future__ import annotations

from dataclasses import dataclass

from ..runtime.requests import RuntimeRequestContext
from ..runtime.responses import PolicyDecision


@dataclass(frozen=True, slots=True)
class PreflightTarget:
    """Bind a normalized URL to its scrape-level policy decision and context."""

    url: str
    decision: PolicyDecision
    request_context: RuntimeRequestContext

    def __post_init__(self) -> None:
        normalized_url = str(self.url or "").strip()
        if not normalized_url:
            raise ValueError("url is required")
        object.__setattr__(self, "url", normalized_url)
