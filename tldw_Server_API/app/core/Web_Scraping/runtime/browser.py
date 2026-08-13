"""Contract-only browser runtime boundaries for later Web_Scraping phases."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol


def _normalize_viewport_dimension(value: Any, *, field_name: str) -> int:
    """Normalize a positive integral viewport dimension."""
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


class RuntimeBrowserRequest(Protocol):
    @property
    def url(self) -> str:
        raise NotImplementedError

    @property
    def resource_type(self) -> str:
        raise NotImplementedError


class RuntimeBrowserRoute(Protocol):
    @property
    def request(self) -> RuntimeBrowserRequest:
        raise NotImplementedError

    async def abort(self) -> None:
        raise NotImplementedError

    async def continue_(self) -> None:
        raise NotImplementedError


class RuntimeWebSocketRoute(Protocol):
    @property
    def url(self) -> str:
        raise NotImplementedError

    def connect_to_server(self) -> Awaitable[Any] | Any:
        raise NotImplementedError

    async def close(
        self,
        *,
        code: int | None = None,
        reason: str | None = None,
    ) -> None:
        raise NotImplementedError


class RuntimeBrowserCDPSession(Protocol):
    async def send(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def on(
        self,
        event: str,
        handler: Callable[[dict[str, Any]], None],
    ) -> None:
        raise NotImplementedError

    async def detach(self) -> None:
        raise NotImplementedError


class RuntimeBrowserLocator(Protocol):
    def nth(self, index: int) -> RuntimeBrowserLocator:
        raise NotImplementedError

    async def count(self) -> int:
        raise NotImplementedError

    async def is_visible(self) -> bool:
        raise NotImplementedError


class RuntimeBrowserPage(Protocol):
    async def goto(self, url: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def reload(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def wait_for_load_state(self, state: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def wait_for_timeout(self, timeout_ms: float) -> Any:
        raise NotImplementedError

    async def content(self) -> str:
        raise NotImplementedError

    async def evaluate(self, expression: str, argument: Any = None) -> Any:
        raise NotImplementedError

    def locator(self, selector: str) -> RuntimeBrowserLocator:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


class RuntimeBrowserContext(Protocol):
    async def route(
        self,
        pattern: str,
        handler: Callable[[RuntimeBrowserRoute], Awaitable[None]],
    ) -> None:
        raise NotImplementedError

    async def route_web_socket(
        self,
        pattern: str,
        handler: Callable[[RuntimeWebSocketRoute], Awaitable[None]],
    ) -> None:
        raise NotImplementedError

    async def add_init_script(self, *, script: str) -> None:
        raise NotImplementedError

    def on(
        self,
        event: str,
        handler: Callable[[RuntimeBrowserRequest], None],
    ) -> None:
        raise NotImplementedError

    async def new_page(self) -> RuntimeBrowserPage:
        raise NotImplementedError

    async def new_cdp_session(
        self,
        page: RuntimeBrowserPage,
    ) -> RuntimeBrowserCDPSession:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


class RuntimeBrowserLauncher(Protocol):
    async def new_context(self, options: BrowserLaunchOptions) -> RuntimeBrowserContext:
        raise NotImplementedError


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
