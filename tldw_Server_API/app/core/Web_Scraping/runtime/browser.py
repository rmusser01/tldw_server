"""Contract-only browser runtime boundaries for later Web_Scraping phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


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

    @property
    def viewport(self) -> dict[str, int]:
        return {"width": int(self.viewport_width), "height": int(self.viewport_height)}
