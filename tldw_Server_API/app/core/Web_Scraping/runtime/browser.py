"""Contract-only browser runtime boundaries for later Web_Scraping phases."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol


def _normalize_viewport_dimension(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite integer")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{field_name} must be a finite integer")
        normalized = int(value)
    elif isinstance(value, str):
        try:
            parsed = Decimal(value.strip())
        except InvalidOperation as exc:
            raise ValueError(f"{field_name} must be a finite integer") from exc
        if not parsed.is_finite() or parsed != parsed.to_integral_value():
            raise ValueError(f"{field_name} must be a finite integer")
        normalized = int(parsed)
    else:
        raise ValueError(f"{field_name} must be a finite integer")
    if normalized < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return normalized


class RuntimeBrowserPage(Protocol):
    async def goto(self, url: str, **kwargs: Any) -> Any:
        """Navigate to a URL."""

    async def content(self) -> str:
        """Return current page content."""

    async def close(self) -> None:
        """Close the page."""


class RuntimeBrowserContext(Protocol):
    async def new_page(self) -> RuntimeBrowserPage:
        """Create a page in this context."""

    async def close(self) -> None:
        """Close the context."""


class RuntimeBrowserLauncher(Protocol):
    async def new_context(self, options: "BrowserLaunchOptions") -> RuntimeBrowserContext:
        """Create a browser context with the supplied options."""


@dataclass(frozen=True, slots=True)
class BrowserLaunchOptions:
    """Browser launch/context options captured without launching a browser."""

    headless: bool = True
    user_agent: str | None = None
    viewport_width: int = 1280
    viewport_height: int = 720

    def __post_init__(self) -> None:
        for field_name in ("viewport_width", "viewport_height"):
            normalized = _normalize_viewport_dimension(getattr(self, field_name), field_name=field_name)
            object.__setattr__(self, field_name, normalized)

    @property
    def viewport(self) -> dict[str, int]:
        return {"width": self.viewport_width, "height": self.viewport_height}
