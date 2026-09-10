"""Shared test doubles for chat macro API integration tests."""

from __future__ import annotations

from typing import Any


class FakeJobManager:
    """Record Jobs submissions while preserving the manager's keyword contract."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        """Record and return one synthetic Jobs row."""
        self.created.append(kwargs)
        return {"id": len(self.created), **kwargs}
