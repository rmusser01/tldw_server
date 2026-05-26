from __future__ import annotations

from typing import Any, Protocol


class ProfileStore(Protocol):
    async def get_profile(self, profile_id: str) -> dict[str, Any] | None: ...


class ExternalRegistryStore(Protocol):
    async def list_servers(self) -> list[dict[str, Any]]: ...


class AuditStore(Protocol):
    async def append_event(self, event: dict[str, Any]) -> None: ...
